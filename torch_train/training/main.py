from __future__ import annotations

import argparse
import importlib.util
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from text_encoder.text_encoder import TextEncoder, encode_text_encoder
from datasets import input_pipeline
from diffusion import rectified_flow
from models.dit import build_dit_model
from training import checkpoint as ckpt_lib
from training import optim as optim_lib
from training.parallel import init_distributed, parallelize, load_tp_batch, compile_blocks
from utils.common import itstime, log
from vae.vae import VAE_CONFIGS, encode_images_to_latents, load_vae, scale_latents


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("i1_config", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_config()


def checkpoint_config(config, latent_size, text_embed_dim, text_num_tokens):
    preset = __import__("models.dit", fromlist=["DualStreamDiT_models"]).DualStreamDiT_models[config.model_size]
    mk = dict(config.model_kwargs)
    return dict(
        input_size=latent_size,
        image_resolution=config.image_size,
        patch_size=config.patch_size,
        in_channels=config.in_channels,
        hidden_size=preset["hidden_size"],
        depth=preset["depth"],
        num_heads=preset["num_heads"],
        mlp_ratio=preset["mlp_ratio"],
        text_embed_dim=text_embed_dim,
        text_num_tokens=text_num_tokens,
        rope_theta=mk.get("rope_theta", 10000.0),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--total_steps", type=int, default=None, help="Override config.total_steps.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override config.input.batch_size.")
    parser.add_argument("--log_every", type=int, default=None, help="Override config.log_training_steps.")
    parser.add_argument("--no_save", action="store_true", help="Disable checkpoint saving.")
    parser.add_argument("--fsdp", type=int, default=None, help="Override config.fsdp_axis_size.")
    parser.add_argument("--tp", type=int, default=None, help="Override config.tensor_parallel_size.")
    parser.add_argument("--grad_accum", type=int, default=None, help="Override config.grad_accum_steps.")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient (activation) checkpointing.")
    parser.add_argument("--ckpt_steps", type=int, default=None, help="Override config.ckpt_steps.")
    parser.add_argument("--keep_ckpt_steps", type=int, default=None, help="Override config.keep_ckpt_steps.")
    parser.add_argument("--data_dir", default=None, help="Override the dataset dir (single source, weight 1.0).")
    parser.add_argument("--no_amp", action="store_true", help="Disable bf16 mixed precision (train in fp32).")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume/fine-tune from (overrides config.resume).")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.total_steps is not None:
        config.total_steps = args.total_steps
    if args.batch_size is not None:
        config.input.batch_size = args.batch_size
    if args.log_every is not None:
        config.log_training_steps = args.log_every
    if args.no_save:
        config.save_ckpt = False
    if args.fsdp is not None:
        config.fsdp_axis_size = args.fsdp
    if args.tp is not None:
        config.tensor_parallel_size = args.tp
    if args.grad_accum is not None:
        config.grad_accum_steps = args.grad_accum
    if args.grad_ckpt:
        config.use_grad_ckpt = True
    if args.ckpt_steps is not None:
        config.ckpt_steps = args.ckpt_steps
    if args.keep_ckpt_steps is not None:
        config.keep_ckpt_steps = args.keep_ckpt_steps
    if args.data_dir is not None:
        config.input.data = [(dict(split="train", data_dir=args.data_dir), 1.0)]
    if args.resume is not None:
        config.resume = args.resume
    if args.no_amp:
        config.amp = False
    amp = config.get("amp", True)

    torch.set_float32_matmul_precision("high")

    dist_info = init_distributed(config)
    device = dist_info.device
    if dist_info.is_main:
        log(f"world_size={dist_info.world_size} data={dist_info.data_size} "
            f"fsdp={dist_info.fsdp_size} model={dist_info.model_size}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if args.workdir and dist_info.is_main:
        os.makedirs(args.workdir, exist_ok=True)
    save_ckpt_path = os.path.join(args.workdir, "checkpoint.pt") if args.workdir else None

    train_ds, ntrain = input_pipeline.build_training_dataset(
        config.input, process_index=dist_info.dp_rank, process_count=dist_info.dp_world
    )

    text_encoder_bundle = TextEncoder(
        config, config.text_encoder_type, config.token_len,
        weight_dtype=torch.bfloat16, device=device,
    )
    tokenizer = text_encoder_bundle.tokenizer
    text_encoder = text_encoder_bundle.text_encoder
    token_len = text_encoder_bundle.text_token_len
    text_embed_dim = text_encoder_bundle.hidden_dim

    train_iter = input_pipeline.start_input_iterator(train_ds, tokenizer, token_len)

    vae = load_vae(config, device, dtype=torch.float32)
    vae_channels = VAE_CONFIGS[config.vae_type]["vae_channels"]
    latent_size = config.image_size // VAE_CONFIGS[config.vae_type]["vae_compression_factor"]

    model = build_dit_model(config, latent_size, text_embed_dim, token_len)
    model.init_weights()
    model = model.to(device=device, dtype=torch.float32)
    for name, p in model.named_parameters():
        if any(re.fullmatch(pat, name) for pat in config.freeze_patterns):
            p.requires_grad_(False)
    ckpt_cfg = checkpoint_config(config, latent_size, text_embed_dim, token_len)

    resume_ckpt = None
    resume_path = ckpt_lib.resolve_resume_path(save_ckpt_path, config)
    if resume_path:
        if dist_info.is_main:
            log(f"resuming from {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        ckpt_lib.load_model_weights(model, resume_ckpt, dist_info)

    if dist_info.is_main:
        n_params = sum(p.numel() for p in model.parameters())
        log(f"model params: {n_params/1e6:.1f}M | latent {vae_channels}x{latent_size}x{latent_size}")

    compile_blocks(model)
    model = parallelize(model, dist_info)

    rf_cfg = rectified_flow.RectifiedFlowConfig.from_config(config.transport)
    mu_dtype = torch.bfloat16 if config.mu_dtype == "bfloat16" else torch.float32
    optimizer = optim_lib.Adam(
        model.named_parameters(), lr=config.lr, b1=config.b1, b2=config.b2, eps=config.adam_eps,
        grad_clip_norm=config.grad_clip_norm, freeze_patterns=config.freeze_patterns, mu_dtype=mu_dtype,
    )
    ema = optim_lib.EMA(model, decay_rate=config.ema_decay_rate) if config.use_ema else None

    first_step = 0
    if resume_ckpt is not None:
        first_step = ckpt_lib.load_train_states(
            ema, optimizer, resume_ckpt, dist_info,
            permuted_keys=getattr(model, "_tp_permuted_keys", set()),
        )
        del resume_ckpt

    total_steps = config.total_steps
    grad_accum = config.grad_accum_steps
    global_bs = config.input.batch_size
    if global_bs % dist_info.dp_world != 0:
        raise ValueError(f"global batch size {global_bs} must be divisible by dp_world {dist_info.dp_world}")
    per_rank_bs = global_bs // dist_info.dp_world
    if per_rank_bs % grad_accum != 0:
        raise ValueError(f"per-rank batch size {per_rank_bs} must be divisible by grad_accum_steps {grad_accum}")
    micro_bs = per_rank_bs // grad_accum

    use_wandb = bool(config.wandb.log_wandb) and dist_info.is_main
    if use_wandb:
        import wandb
        wandb.init(project=str(config.wandb.project), name=str(config.wandb.experiment))
        wandb.config.update(dict(total_steps=total_steps, global_bs=global_bs))

    if dist_info.is_main:
        log(f"training for {total_steps} steps | global_bs={global_bs} "
            f"micro_bs/rank={micro_bs} grad_accum={grad_accum}")

    model.train()
    t_start = time.time()
    if first_step and dist_info.is_main:
        log(f"resumed at step {first_step}")
    torch.manual_seed(config.seed + 1 + dist_info.dp_rank)
    np.random.seed(config.seed + 1 + dist_info.dp_rank)
    batch_specs = [
        ("image", (per_rank_bs, config.image_size, config.image_size, 3), torch.float32),
        ("input_ids", (per_rank_bs, token_len), torch.long),
        ("attention_mask", (per_rank_bs, token_len), torch.long),
    ]
    for step in range(first_step + 1, total_steps + 1):
        batch = load_tp_batch(train_iter, dist_info, batch_specs, device)
        is_log_step = step % config.log_training_steps == 0
        step_loss = torch.zeros((), device=device)
        for micro in range(grad_accum):
            sl = slice(micro * micro_bs, (micro + 1) * micro_bs)
            images = batch["image"][sl].to(device, non_blocking=True)
            input_ids = batch["input_ids"][sl].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"][sl].to(device, non_blocking=True)

            with torch.no_grad():
                latents = encode_images_to_latents(vae, images)
                latents = scale_latents(latents, config).float()
                enc_hidden = encode_text_encoder(text_encoder, input_ids, attention_mask)

            xt, ut, t = rectified_flow.prepare_rectified_flow_inputs(latents, rf_cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                pred = model(xt, t, enc_hidden, attention_mask, train=True)
                loss = F.mse_loss(pred.float(), ut.float())
            if dist_info.is_distributed and grad_accum > 1 and hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(micro == grad_accum - 1)
            (loss / grad_accum).backward()
            step_loss += loss.detach() / grad_accum

        optimizer.step(compute_update_norm=is_log_step)
        for _, p in optimizer.params:
            p.grad = None
        if ema is not None:
            ema.update(model)

        if is_log_step:
            if dist_info.is_distributed:
                torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
            loss_val = step_loss.item()
            l2_params = optim_lib.global_l2_norm(p for p in model.parameters())
            l2_ema = optim_lib.global_l2_norm(ema.shadow.values()) if ema is not None else 0.0
            metrics = {
                "training_loss": loss_val,
                "l2_grads": optimizer.last_grad_norm,
                "l2_updates": optimizer.last_update_norm,
                "l2_params": l2_params,
                "l2_ema_params": l2_ema,
            }
            if dist_info.is_main:
                imgs_per_s = config.log_training_steps * global_bs / (time.time() - t_start)
                log(f"step {step}/{total_steps} loss {loss_val:.5f} "
                    f"l2_grads {metrics['l2_grads']:.3f} l2_updates {metrics['l2_updates']:.2e} "
                    f"imgs/s {imgs_per_s:.0f}")
                if use_wandb:
                    wandb.log({**metrics, "imgs_per_s": imgs_per_s}, step=step)
            t_start = time.time()

        if save_ckpt_path and config.save_ckpt:
            keep_ckpt_steps = config.get("keep_ckpt_steps", None)
            save_now = itstime(step, config.ckpt_steps, total_steps) or (
                keep_ckpt_steps and itstime(step, keep_ckpt_steps, total_steps))
            if save_now:
                copy_step = step if (keep_ckpt_steps and itstime(step, keep_ckpt_steps, total_steps)) else None
                ckpt_lib.save_checkpoint(save_ckpt_path, model, ema, optimizer, step, ckpt_cfg, dist_info,
                                         step_copy=copy_step)

    if dist_info.is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
