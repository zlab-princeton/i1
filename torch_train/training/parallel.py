from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    data_size: int = 1
    fsdp_size: int = 1
    model_size: int = 1
    dp_rank: int = 0
    dp_world: int = 1
    device: torch.device = torch.device("cuda:0")
    mesh: object = None
    tp_group: object = None

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(config) -> DistInfo:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return DistInfo(device=device)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    model_size = int(config.tensor_parallel_size)
    fsdp_size = int(config.fsdp_axis_size)
    if world_size % (model_size * fsdp_size) != 0:
        raise ValueError(f"world_size {world_size} not divisible by fsdp*model ({fsdp_size*model_size})")
    data_size = world_size // (model_size * fsdp_size)

    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh(
        "cuda",
        (data_size, fsdp_size, model_size),
        mesh_dim_names=("data", "fsdp", "model"),
    )
    return DistInfo(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        data_size=data_size,
        fsdp_size=fsdp_size,
        model_size=model_size,
        dp_rank=rank // model_size,
        dp_world=world_size // model_size,
        device=torch.device(f"cuda:{local_rank}"),
        mesh=mesh,
        tp_group=(mesh["model"].get_group() if model_size > 1 else None),
    )


def load_tp_batch(train_iter, dist_info, batch_specs, device):
    if dist_info.model_size == 1:
        return next(train_iter)
    is_leader = dist_info.rank % dist_info.model_size == 0
    src = (dist_info.rank // dist_info.model_size) * dist_info.model_size
    batch = next(train_iter) if is_leader else None
    out = {}
    for key, shape, dtype in batch_specs:
        if is_leader:
            t = batch[key].to(device=device, dtype=dtype).contiguous()
        else:
            t = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(t, src=src, group=dist_info.tp_group)
        out[key] = t
    return out


def _all_blocks(model):
    if model.in_blocks is not None:
        return list(model.in_blocks) + [model.mid_block] + list(model.out_blocks)
    return list(model.blocks)


def compile_blocks(model):
    for block in _all_blocks(model):
        block.compile()
    return model


def _permute_gate_up(w: torch.Tensor, tp: int) -> torch.Tensor:
    two_hf = w.shape[0]
    hf = two_hf // 2
    rest = w.shape[1:]
    w = w.view(2, tp, hf // tp, *rest).permute(1, 0, *range(2, 2 + 1 + len(rest)))
    return w.reshape(two_hf, *rest).contiguous()


def _unpermute_gate_up(w: torch.Tensor, tp: int) -> torch.Tensor:
    two_hf = w.shape[0]
    hf = two_hf // 2
    rest = w.shape[1:]
    w = w.view(tp, 2, hf // tp, *rest).permute(1, 0, *range(2, 2 + 1 + len(rest)))
    return w.reshape(two_hf, *rest).contiguous()


def _apply_tensor_parallel(model, tp_mesh):
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    tp = tp_mesh.size()
    permuted_mlp_ids = set()

    def parallelize_block(block):
        plan = {}
        for prefix in ("mlp_image", "mlp_text"):
            mlp = getattr(block, prefix, None)
            if mlp is None:
                continue
            if hasattr(mlp, "w12"):
                with torch.no_grad():
                    mlp.w12.weight.copy_(_permute_gate_up(mlp.w12.weight.data, tp))
                    mlp.w12.bias.copy_(_permute_gate_up(mlp.w12.bias.data, tp))
                permuted_mlp_ids.add(id(mlp))
                plan[f"{prefix}.w12"] = ColwiseParallel()
                plan[f"{prefix}.w3"] = RowwiseParallel()
            else:
                plan[f"{prefix}.fc1"] = ColwiseParallel()
                plan[f"{prefix}.fc2"] = RowwiseParallel()
        parallelize_module(block, tp_mesh, plan)

    for block in _all_blocks(model):
        parallelize_block(block)
    permuted_keys = set()
    for name, module in model.named_modules():
        if id(module) in permuted_mlp_ids:
            permuted_keys.add(f"{name}.w12.weight")
            permuted_keys.add(f"{name}.w12.bias")
    model._tp_permuted_keys = permuted_keys
    model._tp_size = tp
    return model


def _apply_fsdp(model, dp_mesh):
    from torch.distributed.fsdp import fully_shard

    for block in _all_blocks(model):
        fully_shard(block, mesh=dp_mesh)
    fully_shard(model, mesh=dp_mesh)
    return model


def parallelize(model, dist_info: DistInfo):
    if not dist_info.is_distributed:
        return model
    mesh = dist_info.mesh
    if dist_info.model_size > 1:
        _apply_tensor_parallel(model, mesh["model"])
    if dist_info.data_size * dist_info.fsdp_size > 1:
        dp_mesh = mesh["data", "fsdp"]
        _apply_fsdp(model, dp_mesh)
    return model
