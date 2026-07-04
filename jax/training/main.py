import os
import jax
import jax.numpy as jnp
if not jax.distributed.is_initialized():
    try:
        jax.distributed.initialize()
    except Exception as e:
        pass
import functools
import itertools

import flax
import optax
import numpy as np
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import tensorflow as tf

from clu import parameter_overview
from tensorflow.io import gfile
import multiprocessing.pool
from jax.experimental import mesh_utils
from jax.experimental import multihost_utils
from jax.experimental import pjit
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from datasets import input_pipeline
from diffusion import rectified_flow
from training import optim
from training.inprocess_auto_generate import InProcessAutoGenerator
from utils import common as utils
from utils.common import (
    MetricWriter,
    checkpointing_timeout,
    chrono,
    load_checkpoint,
    recover_dtype,
    save_checkpoint,
    sync,
)
from text_encoder.text_encoder import TextEncoder, encode_text_encoder
from vae.vae import VAE_CONFIGS, load_vae, scale_latents
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False
from models.cross_attn_backbone import (
    CrossAttnDiT,
    CrossAttnDiTConfig,
    CrossAttnDiT_models,
)
from models.single_stream_backbone import (
    SingleStreamDiT,
    SingleStreamDiTConfig,
    SingleStreamDiT_models,
)
from models.dual_stream_backbone import (
    DualStreamDiT,
    DualStreamDiTConfig,
    DualStreamDiT_models,
)
from training.optim import replace_frozen, steps

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Path to save checkpoints and generated images (if auto_generate_on_ckpt is turned on).")
flags.DEFINE_bool("auto_generate_on_ckpt", default=False, help="Run in-process generation + zip + upload during training whenever a permanent checkpoint is saved.")
flags.DEFINE_integer("auto_generate_per_proc_batch_size", default=256, help="Per-process batch size for in-process generation")
flags.DEFINE_bool("check_gcs_region", default=True, help="Check that GCS buckets are in the same region as the TPU.")

jax.config.parse_flags_with_absl()


def get_weight_dtype(precision_value):
    if precision_value == "fp16":
        return jnp.float16
    if precision_value == "bf16":
        return jnp.bfloat16
    return jnp.float32


def cast_tree_to_dtype(tree, dtype):
    if dtype == jnp.float32:
        return tree

    def _cast(value):
        if hasattr(value, "dtype") and jnp.issubdtype(value.dtype, jnp.floating):
            return value.astype(dtype)
        return value

    return jax.tree_util.tree_map(_cast, tree)


def _ema_update(avg_params, new_params, decay_rate=0.99):
    """Applies an exponential moving average."""
    decay = jnp.asarray(decay_rate, jnp.float32)

    def _weighted_average(p1, p2):
        # Keep EMA in float32 to avoid decay rounding when model weights are bf16.
        p1_f32 = jnp.asarray(p1, jnp.float32)
        p2_f32 = jnp.asarray(p2, jnp.float32)
        return (1.0 - decay) * p1_f32 + decay * p2_f32

    return jax.tree_util.tree_map(_weighted_average, new_params, avg_params)


_BACKBONE_MODELS = {
    "cross_attn": (
        CrossAttnDiTConfig,
        CrossAttnDiT,
        CrossAttnDiT_models,
    ),
    "single_stream": (
        SingleStreamDiTConfig,
        SingleStreamDiT,
        SingleStreamDiT_models,
    ),
    "dual_stream": (
        DualStreamDiTConfig,
        DualStreamDiT,
        DualStreamDiT_models,
    ),
}

def build_dit_model(config, latent_size, text_embed_dim, text_num_tokens, model_dtype):
    config_cls, model_cls, presets = _BACKBONE_MODELS[config.backbone]
    preset = presets[config.model_size]
    model_kwargs = dict(config.model_kwargs)
    model_kwargs.setdefault("in_channels", VAE_CONFIGS[config.vae_type]["vae_channels"])
    return model_cls(
        config_cls(
            input_size=latent_size,
            image_resolution=config.image_size,
            patch_size=config.patch_size,
            text_embed_dim=text_embed_dim,
            text_num_tokens=text_num_tokens,
            use_qknorm=config.use_qknorm,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
            use_grad_ckpt=config.use_grad_ckpt,
            **preset,
            **model_kwargs,
        ),
        dtype=model_dtype,
    )

def split_key_array(key_array, num: int):
    flat = key_array.reshape((-1, 2))
    split_flat = jax.vmap(lambda k: jax.random.split(k, num))(flat)
    return split_flat.reshape(key_array.shape[:-1] + (num, 2))

def _extract_latent_dist(encoded):
    if hasattr(encoded, "latent_dist"):
        return encoded.latent_dist
    if isinstance(encoded, dict) and "latent_dist" in encoded:
        return encoded["latent_dist"]
    return encoded


def _sample_latents(latent_dist, rng):
    return latent_dist.sample(rng) if hasattr(latent_dist, "sample") else latent_dist


def encode_images_to_latents(jax_vae_model, vae_params, images, rng, config, vae_dtype):
    vae_config = VAE_CONFIGS[config.vae_type]
    pretrained_vae_name_or_path = vae_config["pretrained_vae_name_or_path"]
    vae_channels = vae_config["vae_channels"]
    if pretrained_vae_name_or_path == "Qwen/Qwen-Image":
        images = jnp.transpose(images, (0, 3, 1, 2))
        images = images[:, :, None, :, :]
    else:
        images = jnp.transpose(images, (0, 3, 1, 2))

    images = images.astype(vae_dtype)
    encoded = jax_vae_model.apply(
        {"params": vae_params},
        images,
        deterministic=True,
        method=jax_vae_model.encode,
    )
    latent_dist = _extract_latent_dist(encoded)
    latents = _sample_latents(latent_dist, rng)

    if pretrained_vae_name_or_path == "Qwen/Qwen-Image" and latents.ndim == 5:
        latents = latents[:, :, 0, :, :]

    if latents.ndim == 4:
        channels_first = latents.shape[1] == vae_channels
        channels_last = latents.shape[-1] == vae_channels
        if channels_last and not channels_first:
            latents = jnp.transpose(latents, (0, 3, 1, 2))
        elif channels_last and channels_first:
            latents = jnp.transpose(latents, (0, 3, 1, 2))

    return latents


def _choose_cross_worker_ici_mesh_shape(axis_sizes, process_count: int):
    """
    E.g., single-slice TPU v6-128 with TP=4, axis_sizes=(128/FSDP/TP, FSDP=1, TP=4)=(32, 1, 4), process_count=num_hosts=128/4=32.
    The (32, 1, 4) parallelism is handled by cross-worker ICI and within-worker ICI.
    Cross-worker ICI always contributes a factor of process_count=32 to the parallelism.
    Here we want to decide how much of process_count=32 to allocate to data parallel, FSDP, and TP.
    We look for (d_data, d_fsdp, d_model) under the following criteria:
    (1) d_data * d_fsdp * d_model == process_count
    (2) 128/FSDP/TP % d_data = 0; FSDP % d_fsdp = 0; TP % d_model = 0
    (3) In descending order of priority to maximize: d_data > d_fsdp > d_model
    """
    data_axis_size, fsdp_axis_size, model_axis_size = axis_sizes
    for d_data in [i for i in range(data_axis_size, 0, -1) if data_axis_size % i == 0]:
        if process_count % d_data != 0:
            continue
        remaining = process_count // d_data
        for d_fsdp in [i for i in range(fsdp_axis_size, 0, -1) if fsdp_axis_size % i == 0]:
            if remaining % d_fsdp != 0:
                continue
            d_model = remaining // d_fsdp
            if model_axis_size % d_model != 0:
                continue
            return (d_data, d_fsdp, d_model)

def create_data_fsdp_model_mesh(fsdp_axis_size: int, model_axis_size: int) -> Mesh:
    shards_per_data = fsdp_axis_size * model_axis_size
    assert jax.device_count() % shards_per_data == 0, f"Device count {jax.device_count()} must be divisible by fsdp*model ({shards_per_data})"
    data_axis_size = jax.device_count() // shards_per_data
    if jax.process_count() == 1:
        devices = mesh_utils.create_device_mesh((data_axis_size, fsdp_axis_size, model_axis_size))
    else:
        dcn_mesh_shape = _choose_cross_worker_ici_mesh_shape(
            (data_axis_size, fsdp_axis_size, model_axis_size),
            jax.process_count(),
        )
        """
        E.g., single-slice TPU v6-128 with TP=4, axis_sizes=(128/FSDP/TP, FSDP=1, TP=4)=(32, 1, 4), process_count=num_hosts=128/4=32.
        In _choose_cross_worker_ici_mesh_shape, we've decided how to split the number of parallel workers to satisfy axis_sizes=(128/FSDP/TP, FSDP=1, TP=4)=(32, 1, 4).
        _choose_cross_worker_ici_mesh_shape outputs a tuple (d_data, d_fsdp, d_model).
        Now the remaining dimensions ((128/FSDP/TP) / d_data, FSDP / d_fsdp, TP / d_model) should be handled by within-worker ICI.
        """
        local_mesh_shape = tuple(
            axis_size // dcn_axis_size
            for axis_size, dcn_axis_size in zip(
                (data_axis_size, fsdp_axis_size, model_axis_size),
                dcn_mesh_shape,
            )
        )
        """
        Build a hybrid (cross-worker ICI x within-worker ICI) mesh. Although
        create_hybrid_device_mesh names the outer shape dcn_mesh_shape, with
        process_is_granule=True each host/JAX process is treated as one outer
        granule. Within each process we create an ICI mesh. This guarantees
        each host's devices are a contiguous subcube of the global mesh, which
        is required by multihost_utils.host_local_array_to_global_array.
        """
        devices = mesh_utils.create_hybrid_device_mesh(
            mesh_shape=local_mesh_shape,
            dcn_mesh_shape=dcn_mesh_shape,
            process_is_granule=True,
        )
    return Mesh(devices, ("data", "fsdp", "model"))


def make_batch_spec(example_batch):
    return jax.tree_util.tree_map(
        lambda x: PartitionSpec(("data", "fsdp"), *([None] * (x.ndim - 1))), example_batch
    )


def shard_tree_with_named_spec(tree, spec_tree, mesh: Mesh):
    """Shard a pytree according to PartitionSpecs, even on multi-host meshes."""

    def _to_array(x, spec):
        if spec is None:
            return x
        if not hasattr(x, "shape"):
            return x
        arr = np.asarray(x)
        sharding = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(
            arr.shape,
            sharding,
            lambda idx: arr[idx],
        )

    with mesh:
        return jax.tree_util.tree_map(_to_array, tree, spec_tree)


def default_param_pspecs(
    params,
    *,
    model_axis: str = "model",
    fsdp_axis: str = "fsdp",
    allow_separate_model_and_fsdp_axes: bool = True,
):
    """
    Following the example at:
    https://github.com/AI-Hypercomputer/maxtext/blob/d5ea751920732be26f26686015be8888fec44055/pedagogical_examples/shardings.py
    """
    def _spec(value):
        if not hasattr(value, "ndim"):
            return PartitionSpec()
        if value.ndim == 2:  # dense kernel: (in, out)
            if allow_separate_model_and_fsdp_axes:
                return PartitionSpec(model_axis, fsdp_axis)
            return PartitionSpec((model_axis, fsdp_axis), None)
        if value.ndim == 4:  # conv kernel: (kh, kw, in, out)
            if allow_separate_model_and_fsdp_axes:
                return PartitionSpec(None, None, model_axis, fsdp_axis)
            return PartitionSpec(None, None, (model_axis, fsdp_axis), None)
        return PartitionSpec()

    pspecs = jax.tree_util.tree_map(_spec, params)
    if isinstance(pspecs, flax.core.FrozenDict):
        return flax.core.unfreeze(pspecs)
    return pspecs


def materialize_tree_for_checkpoint(tree):
    """Converts a pytree to host arrays, gathering multi-host global arrays."""

    def _to_host(x):
        if isinstance(x, jax.Array):
            if x.is_fully_addressable:
                return np.asarray(x)
            # Multi-host global arrays cannot be fetched directly on one host.
            return multihost_utils.process_allgather(x)
        return x

    return jax.tree_util.tree_map(_to_host, tree)


def _get_data_dirs(input_config):
    return [dataset_cfg.get("data_dir") for dataset_cfg, _ in input_config.data]

def main(argv):
    del argv
    tf.config.experimental.set_visible_devices([], "GPU")
    config = flags.FLAGS.config
    workdir = flags.FLAGS.workdir

    if flags.FLAGS.check_gcs_region:
        import requests
        from google.cloud import storage
        client = storage.Client()
        METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
        HEADERS = {"Metadata-Flavor": "Google"}
        tpu_zone_path = requests.get(METADATA_URL, headers=HEADERS, timeout=20).text
        tpu_zone = tpu_zone_path.split("/")[-1]
        tpu_region = "-".join(tpu_zone.split("-")[:-1])
        workdir_bucket_name = workdir.removeprefix("gs://").split("/")[0]
        workdir_bucket = client.get_bucket(workdir_bucket_name)
        workdir_bucket_region = workdir_bucket.location.lower()
        logging.info(_get_data_dirs(config.input))
        assert len(_get_data_dirs(config.input)) > 0
        for data_dir in _get_data_dirs(config.input):
            bucket_name = data_dir.removeprefix("gs://").split("/")[0]
            bucket = client.get_bucket(bucket_name)
            bucket_region = bucket.location.lower()
            if (bucket_region is None) or (len(bucket_region) < 8) or (tpu_region is None) or (len(tpu_region) < 8) or (workdir_bucket_region is None) or (len(workdir_bucket_region) < 8):
                logging.info(f"One of them is empty: TPU region {tpu_region}, bucket region {bucket_region}, workdir bucket region {workdir_bucket_region}")
                exit()
            elif bucket_region != tpu_region:
                logging.info(f"TPU region {tpu_region} doesn't match bucket region {bucket_region}")
                exit()
            elif workdir_bucket_region != tpu_region:
                logging.info(f"TPU region {tpu_region} doesn't match workdir bucket region {workdir_bucket_region}")
                exit()
            else:
                logging.info(f"TPU region {tpu_region}, bucket region {bucket_region}, workdir bucket region {workdir_bucket_region}, proceed")

    model_dtype = jnp.float32
    text_encoder_dtype = get_weight_dtype(config.text_encoder_precision)

    rectified_flow_cfg = rectified_flow.RectifiedFlowConfig.from_config(config.transport)

    logging.info(
        f"\u001b[33mHello from process {jax.process_index()} holding "
        f"{jax.local_device_count()}/{jax.device_count()} devices and "
        f"writing to workdir {workdir}.\u001b[0m")
    if config.wandb.log_wandb:
        if has_wandb and jax.process_index() == 0:
            if config.wandb.wandb_offline:
                os.environ["WANDB_MODE"] = 'offline'
            else:
                wandb.init(project=str(config.wandb.project), name=str(config.wandb.experiment), resume=config.wandb.resume)
                wandb.config.update(dict(config))
        else:
            logging.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    save_ckpt_path = None
    if workdir:
        gfile.makedirs(workdir)
        save_ckpt_path = os.path.join(workdir, "checkpoint.npz")
    auto_generator = InProcessAutoGenerator(
        enabled=flags.FLAGS.auto_generate_on_ckpt,
        train_config=config,
        save_ckpt_path=save_ckpt_path,
        per_proc_batch_size=flags.FLAGS.auto_generate_per_proc_batch_size,
    )
    pool = multiprocessing.pool.ThreadPool()
    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    def info(s, *a):
        logging.info("\u001b[33mNOTE\u001b[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    write_note("Initializing...")

    batch_size = config.input.batch_size
    tensor_parallelism = int(config.tensor_parallel_size)
    fsdp_axis_size = int(config.fsdp_axis_size)
    mesh = create_data_fsdp_model_mesh(fsdp_axis_size, tensor_parallelism)
    data_axis_size, fsdp_axis_size, model_axis_size = mesh.devices.shape
    total_batch_shards = data_axis_size * fsdp_axis_size
    if batch_size % total_batch_shards != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by data*fsdp axis size ({total_batch_shards})"
        )
    per_data_batch = batch_size // total_batch_shards
    if per_data_batch % config.grad_accum_steps != 0:
        raise ValueError(
            f"Per-shard batch size ({per_data_batch}) must be divisible by grad_accum_steps ({config.grad_accum_steps})"
        )
    info(
        "Global batch size %d on %d hosts results in %d local batch size. Mesh: data=%d, fsdp=%d, model=%d.",
        batch_size,
        jax.process_count(),
        batch_size // jax.process_count(),
        data_axis_size,
        fsdp_axis_size,
        model_axis_size,
    )

    metric = MetricWriter(workdir, config)
    write_note("Initializing train dataset...")

    train_ds, ntrain_img = input_pipeline.build_training_dataset(config.input)
    text_encoder_bundle = TextEncoder(
        config,
        config.text_encoder_type,
        config.token_len,
        weight_dtype=text_encoder_dtype,
    )
    tokenizer, text_encoder, text_encoder_params = text_encoder_bundle.tokenizer, text_encoder_bundle.text_encoder, text_encoder_bundle.text_encoder_params
    text_encoder_params = cast_tree_to_dtype(text_encoder_params, text_encoder_dtype)
    is_multi_text_encoder = isinstance(text_encoder, (list, tuple))
    jax_vae_model, vae_params = load_vae(config)
    vae_input_dtype = getattr(jax_vae_model, "dtype", jax.tree_util.tree_leaves(vae_params)[0].dtype)

    n_prefetch = config.prefetch_to_device
    train_iter = input_pipeline.start_tokenize_input_iterator(train_ds, n_prefetch, tokenizer=tokenizer)
    first_batch_local = next(train_iter)
    batch_spec = make_batch_spec(first_batch_local)
    # put the first batch back
    train_iter = itertools.chain([first_batch_local], train_iter)
    total_steps = steps("total", config, ntrain_img, batch_size)
    def get_steps(name, default=ValueError, cfg=config):
        return steps(name, cfg, ntrain_img, batch_size, total_steps, default)

    chrono.inform(total_steps=total_steps, global_bs=batch_size,
                    steps_per_epoch=ntrain_img / batch_size,
                    measure=metric.measure, write_note=write_note)

    info("Running for %d steps, that means %f epochs",
        total_steps, total_steps * batch_size / ntrain_img)

    vae_config = VAE_CONFIGS[config.vae_type]
    latent_size = config.image_size//vae_config["vae_compression_factor"]
    model = build_dit_model(
        config,
        latent_size,
        text_encoder_bundle.hidden_dim,
        text_encoder_bundle.text_token_len,
        model_dtype,
    )

    def _dit_forward(params, x_latents, t_steps, enc_h, attn_mask, rng_drop_path, rng_drop_text):
        return model.apply(
            {"params": params},
            x_latents, t_steps, enc_h,
            mask=attn_mask,
            train=True,
            rngs={"drop_path": rng_drop_path, "drop_text": rng_drop_text},
    )

    @functools.partial(jax.jit, backend="cpu", static_argnums=(1,2))
    def init(rng, token_len, text_encoder_hidden_dim):
        x = jnp.zeros((2, vae_config["vae_channels"], latent_size, latent_size), model_dtype)
        t = jnp.linspace(0, 1000, 2, dtype=model_dtype)
        if isinstance(token_len, (list, tuple)):
            assert len(token_len) == len(text_encoder_hidden_dim)
            y = [jnp.zeros((2, cur_len, cur_dim), dtype=model_dtype) for cur_len, cur_dim in zip(token_len, text_encoder_hidden_dim)]
            mask = jnp.ones((2, int(sum(token_len))), dtype=model_dtype)
        else:
            y = jnp.zeros((2, token_len, text_encoder_hidden_dim), dtype=model_dtype)
            mask = jnp.ones((2, token_len), dtype=model_dtype)
        params = flax.core.unfreeze(model.init(rng, x, t, y, mask=mask))["params"]
        return params
    rng, rng_init = jax.random.split(rng)
    init_rngs = {'params': rng_init}

    with chrono.log_timing("z/secs/init"):
        init_token_lens = tuple(text_encoder_bundle.text_token_len) if is_multi_text_encoder else text_encoder_bundle.text_token_len
        init_embed_dims = tuple(text_encoder_bundle.hidden_dim) if is_multi_text_encoder else text_encoder_bundle.hidden_dim
        params_cpu = init(init_rngs, init_token_lens, init_embed_dims)

    params_cpu = cast_tree_to_dtype(params_cpu, jnp.float32)
    params_ema_cpu = params_cpu
    if jax.process_index() == 0:
        num_params = sum(p.size for p in jax.tree_util.tree_leaves(params_cpu))
        parameter_overview.log_parameter_overview(params_cpu, msg="init params")
        metric.measure("num_params", num_params)
    write_note(f"Initializing {config.optax_name} optimizer...")
    tx, sched_fns = optim.make(config, params_cpu)
    opt_cpu = jax.jit(tx.init, backend="cpu")(params_cpu)
    sched_fns_cpu = [jax.jit(sched_fn, backend="cpu") for sched_fn in sched_fns]
    def _extract_pos_embed(tree):
        pos_leaves = {}
        for name, val in utils.tree_flatten_with_names(tree)[0]:
            if "pos_embed" in name:
                pos_leaves[name] = val
        return pos_leaves

    init_pos_embed_params = _extract_pos_embed(params_cpu)
    init_pos_embed_opt = _extract_pos_embed(opt_cpu)

    use_x_prediction = False
    t_eps = 0.05
    prediction_mode = getattr(rectified_flow_cfg, "prediction", "velocity")
    use_x_prediction = str(prediction_mode).lower() in ("x", "sample", "data")

    def _to_velocity(pred, sample, t_vec):
        if not use_x_prediction:
            return pred
        t_vec = t_vec.astype(pred.dtype)
        t_broadcast = t_vec.reshape((t_vec.shape[0],) + (1,) * (pred.ndim - 1))
        eps = jnp.asarray(t_eps, dtype=pred.dtype)
        denom = jnp.maximum(1.0 - t_broadcast, eps)
        return (pred - sample) / denom

    def update_fn(params, ema_params, text_encoder_params, vae_params, opt, batch, train_rng):
        measurements = {}

        params_fwd = cast_tree_to_dtype(params, model_dtype)

        def loss_fn_spmd(params, mbatch, rng_s, rng_dp, rng_dt, text_encoder_params, vae_params):
            rng_sample_latents = jax.random.fold_in(rng_s, 0)
            rng_noise_local = jax.random.fold_in(rng_s, 1)
            rng_timestep_local = jax.random.fold_in(rng_s, 2)
            latents = encode_images_to_latents(
                jax_vae_model,
                vae_params,
                mbatch["image"],
                rng_sample_latents,
                config,
                vae_input_dtype,
            )
            latents = jax.lax.stop_gradient(latents)
            latents = scale_latents(latents, config)
            latents_f32 = latents.astype(jnp.float32)

            xt, ut, t = rectified_flow.prepare_rectified_flow_inputs(
                latents,
                rng_noise_local,
                rng_timestep_local,
                rectified_flow_cfg,
            )
            model_latents = xt
            timesteps = t
            target = ut

            # Text encoder (no grads)
            if is_multi_text_encoder:
                assert len(text_encoder_bundle.text_encoder_type) == len(text_encoder) and len(text_encoder_bundle.text_encoder_type) == len(text_encoder_params)
                assert isinstance(mbatch["labels"], (list, tuple))
                encoder_hidden_states = []
                masks = []
                drop_prefix_lens = text_encoder_bundle.drop_prefix_token_len
                for enc_type, enc, enc_params, labels, drop_prefix_len in zip(
                    text_encoder_bundle.text_encoder_type,
                    text_encoder,
                    text_encoder_params,
                    mbatch["labels"],
                    drop_prefix_lens,
                ):
                    cur_mask = labels[:, :, 1]
                    cur_hidden = encode_text_encoder(
                        enc,
                        enc_params,
                        enc_type,
                        labels[:, :, 0],
                        cur_mask,
                        drop_prefix_len,
                    )
                    if drop_prefix_len:
                        cur_mask = cur_mask[:, drop_prefix_len:]
                    encoder_hidden_states.append(cur_hidden)
                    masks.append(cur_mask)
                mask = jnp.concatenate(masks, axis=1)
            else:
                mask = mbatch["labels"][:, :, 1]
                encoder_hidden_states = encode_text_encoder(
                    text_encoder,
                    text_encoder_params,
                    text_encoder_bundle.text_encoder_type,
                    mbatch["labels"][:, :, 0],
                    mask,
                    text_encoder_bundle.drop_prefix_token_len,
                )
                if text_encoder_bundle.drop_prefix_token_len:
                    mask = mask[:, text_encoder_bundle.drop_prefix_token_len:]

            model_latents_mp = model_latents.astype(model_dtype)
            timesteps_mp = timesteps.astype(model_dtype)
            encoder_hidden_states_mp = cast_tree_to_dtype(encoder_hidden_states, model_dtype)
            mask_mp = mask.astype(model_dtype) if mask is not None else None

            model_output = _dit_forward(
                params, model_latents_mp, timesteps_mp, encoder_hidden_states_mp, mask_mp, rng_dp, rng_dt
            )

            # Compute losses in float32 for stability.
            model_output_f32 = model_output.astype(jnp.float32)
            model_latents_f32 = model_latents.astype(jnp.float32)
            target_f32 = target.astype(jnp.float32)
            timesteps_f32 = timesteps.astype(jnp.float32)

            velocity_target = target_f32
            if use_x_prediction:
                velocity_target = _to_velocity(latents_f32, model_latents_f32, timesteps_f32)
            velocity_pred = _to_velocity(model_output_f32, model_latents_f32, timesteps_f32)
            loss_mse = jnp.mean((velocity_pred - velocity_target) ** 2)
            return loss_mse


        def local_loss_and_grad(params_fwd, batch, train_rng, text_encoder_params, vae_params):
            split4 = split_key_array(train_rng, 4)
            rng_sample = split4[..., 0, :]
            new_train_rng = split4[..., 1, :]
            rng_drop_path = split4[..., 2, :]
            rng_drop_text = split4[..., 3, :]

            S = config.grad_accum_steps
            rng_sample_m = split_key_array(rng_sample, S)
            rng_drop_path_m = split_key_array(rng_drop_path, S)
            rng_drop_text_m = split_key_array(rng_drop_text, S)

            rng_sample_m = jnp.moveaxis(rng_sample_m, -2, 0)
            rng_drop_path_m = jnp.moveaxis(rng_drop_path_m, -2, 0)
            rng_drop_text_m = jnp.moveaxis(rng_drop_text_m, -2, 0)

            def _get_global_bs(tree):
                for leaf in jax.tree_util.tree_leaves(tree):
                    if hasattr(leaf, "shape") and leaf.ndim >= 1:
                        return int(leaf.shape[0])
                raise ValueError("Batch must contain at least one array leaf with a leading batch dimension.")

            global_bs = _get_global_bs(batch)
            if global_bs % total_batch_shards != 0:
                raise ValueError(
                    f"Global batch ({global_bs}) must be divisible by data*fsdp shards ({total_batch_shards})."
                )
            per_shard_bs = global_bs // total_batch_shards
            if per_shard_bs % config.grad_accum_steps != 0:
                raise ValueError(
                    f"Per-shard batch ({per_shard_bs}) must be divisible by grad_accum_steps ({config.grad_accum_steps})."
                )
            micro_bs_local = per_shard_bs // config.grad_accum_steps

            def _reshape_for_accum(x):
                if not hasattr(x, "shape") or x.ndim == 0:
                    return x
                return x.reshape((total_batch_shards, per_shard_bs) + tuple(x.shape[1:]))

            def _slice_micro(x, i):
                if not hasattr(x, "shape") or x.ndim == 0:
                    return x
                return jax.lax.dynamic_slice_in_dim(x, i * micro_bs_local, micro_bs_local, axis=1)

            def _flatten_micro(x):
                if not hasattr(x, "shape") or x.ndim == 0:
                    return x
                # (data*fsdp, micro_bs, ...) -> (global_micro_bs, ...)
                return x.reshape((-1,) + tuple(x.shape[2:]))

            batch_accum = jax.tree_util.tree_map(_reshape_for_accum, batch)

            def tree_zero_like(ref_tree):
                return jax.tree_util.tree_map(lambda x: jnp.zeros_like(x, dtype=jnp.float32), ref_tree)

            def tree_add(a, b):
                return jax.tree_util.tree_map(lambda x, y: x + y, a, b)

            def body(carry, i):
                grads_acc, loss_acc = carry
                mb = jax.tree_util.tree_map(lambda x: _slice_micro(x, i), batch_accum)
                mb = jax.tree_util.tree_map(_flatten_micro, mb)
                li, gi = jax.value_and_grad(loss_fn_spmd)(
                    params_fwd,
                    mb,
                    rng_sample_m[i],
                    rng_drop_path_m[i],
                    rng_drop_text_m[i],
                    text_encoder_params,
                    vae_params,
                )
                gi = cast_tree_to_dtype(gi, jnp.float32)
                return (tree_add(grads_acc, gi), loss_acc + li), None

            (carry_out, _) = jax.lax.scan(
                body,
                (tree_zero_like(params_fwd), jnp.array(0.0, dtype=jnp.float32)),
                jnp.arange(S),
            )
            grads_sum, loss_sum = carry_out

            inv_S = jnp.asarray(1.0 / S, dtype=jnp.float32)
            grads = jax.tree_util.tree_map(lambda g: g * inv_S, grads_sum)
            l = loss_sum * inv_S
            return grads, l, new_train_rng

        grads, l, new_train_rng = local_loss_and_grad(
            params_fwd, batch, train_rng, text_encoder_params, vae_params
        )

        updates, opt = tx.update(grads, opt, params)
        params = optax.apply_updates(params, updates)

        if config.use_ema:
            ema_params = _ema_update(ema_params, params, decay_rate=config.ema_decay_rate)

        measurements["loss_mse"] = l
        gs = jax.tree_util.tree_leaves(replace_frozen(config.schedule, grads, 0.))
        measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
        ps = jax.tree_util.tree_leaves(params)
        measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
        eps = jax.tree_util.tree_leaves(ema_params)
        measurements["l2_ema_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in eps]))
        us = jax.tree_util.tree_leaves(updates)
        measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

        return params, ema_params, opt, new_train_rng, l, measurements

    resume_ckpt_path = None
    if save_ckpt_path and gfile.exists(save_ckpt_path):
        resume_ckpt_path = save_ckpt_path
    elif config.resume:
        if flags.FLAGS.check_gcs_region:
            resume_bucket_name = config.resume.removeprefix("gs://").split("/")[0]
            resume_bucket = client.get_bucket(resume_bucket_name)
            resume_bucket_region = resume_bucket.location.lower()
            logging.info(_get_data_dirs(config.input))
            assert len(_get_data_dirs(config.input)) > 0
            for data_dir in _get_data_dirs(config.input):
                bucket_name = data_dir.removeprefix("gs://").split("/")[0]
                bucket = client.get_bucket(bucket_name)
                bucket_region = bucket.location.lower()
                if resume_bucket_region != tpu_region:
                    logging.info(f"TPU region {tpu_region} doesn't match resume bucket region {resume_bucket_region}")
                    exit()
                else:
                    logging.info(f"TPU region {tpu_region}, resume bucket region {resume_bucket_region}, proceed")
        resume_ckpt_path = config.resume

    if resume_ckpt_path:
        write_note("Resume training from checkpoint...")
        checkpoint = {
            "params": params_cpu,
            'params_ema': params_cpu,
            "opt": opt_cpu,
            "chrono": chrono.save(),
        }
        checkpoint_tree = jax.tree_util.tree_structure(checkpoint)
        loaded = load_checkpoint(checkpoint_tree, resume_ckpt_path)
        checkpoint = jax.tree_util.tree_map(recover_dtype, loaded)
        params_cpu, params_ema_cpu, opt_cpu = checkpoint["params"], checkpoint["params_ema"], checkpoint["opt"]
        def _replace_pos_embed(loaded_tree, pos_leaves):
            def _maybe_replace(name, loaded_val):
                if name in pos_leaves:
                    return pos_leaves[name]
                return loaded_val
            return utils.tree_map_with_names(_maybe_replace, loaded_tree)
        params_cpu = _replace_pos_embed(params_cpu, init_pos_embed_params)
        params_ema_cpu = _replace_pos_embed(params_ema_cpu, init_pos_embed_params)
        opt_cpu = _replace_pos_embed(opt_cpu, init_pos_embed_opt)
        chrono.load(checkpoint["chrono"])
        keep_ckpt_steps = get_steps("keep_ckpt", None)
        if (
            jax.process_index() == 0
            and save_ckpt_path
            and resume_ckpt_path == save_ckpt_path
            and keep_ckpt_steps
        ):
            resumed_step = int(optim.get_count(opt_cpu))
            if resumed_step > 0 and resumed_step % keep_ckpt_steps == 0:
                keep_ckpt_path = f"{save_ckpt_path}-{resumed_step:09d}"
                if not gfile.exists(keep_ckpt_path):
                    logging.info(
                        "Backfilling missing keep checkpoint copy: %s",
                        keep_ckpt_path,
                    )
                    gfile.copy(save_ckpt_path, keep_ckpt_path, overwrite=True)

    def _to_float32(tree):
        return jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float32) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x,
            tree,
        )
    if config.use_ema and not resume_ckpt_path:
        params_ema_cpu = _to_float32(params_cpu)

    write_note("Kicking off misc stuff...")
    first_step = optim.get_count(opt_cpu)
    if (
        auto_generator.enabled
        and resume_ckpt_path == save_ckpt_path
        and first_step > 0
        and utils.itstime(first_step, get_steps("keep_ckpt", None), total_steps, host=None, first=False)
        and not auto_generator.is_step_completed(first_step)
    ):
        write_note(f"Running missed auto-generation for resumed checkpoint at step {first_step}...")
        auto_generator.run_after_checkpoint(step=first_step, ckpt_writer=None)
    chrono.inform(first_step=first_step)

    param_pspecs = default_param_pspecs(
        params_cpu,
        model_axis="model",
        fsdp_axis="fsdp",
        allow_separate_model_and_fsdp_axes=(fsdp_axis_size == 1 or model_axis_size == 1),
    )
    if config.shard_text_encoder_params:
        write_note("Shard text encoder params")
        text_encoder_pspecs = default_param_pspecs(
            text_encoder_params,
            model_axis="model",
            fsdp_axis="fsdp",
            allow_separate_model_and_fsdp_axes=(fsdp_axis_size == 1 or model_axis_size == 1),
        )
    else:
        # each device holds a full copy of the text encoder params
        text_encoder_pspecs = jax.tree_util.tree_map(lambda _: PartitionSpec(), text_encoder_params)

    params_treedef = jax.tree_util.tree_structure(params_cpu)
    def _is_param_like_tree(x):
        try:
            return jax.tree_util.tree_structure(flax.core.unfreeze(x) if isinstance(x, flax.core.FrozenDict) else x) == params_treedef
        except Exception:
            return False
    def _make_opt_pspec(tree):
        if isinstance(tree, optax.EmptyState):
            return None
        if hasattr(tree, "shape"):
            # scalars/counters or standalone arrays are replicated
            return PartitionSpec()
        if _is_param_like_tree(tree):
            # moments/state shaped like params follow param sharding
            return param_pspecs
        return jax.tree_util.tree_map(_make_opt_pspec, tree)
    opt_pspecs = _make_opt_pspec(opt_cpu)

    update_fn = pjit.pjit(
        update_fn,
        in_shardings=(
            param_pspecs,
            param_pspecs,
            text_encoder_pspecs,
            PartitionSpec(),
            opt_pspecs,
            batch_spec,
            PartitionSpec(),
        ),
        out_shardings=(
            param_pspecs,
            param_pspecs,
            opt_pspecs,
            PartitionSpec(),
            PartitionSpec(),
            None,
        ),
        donate_argnums=(0, 1, 4),
    )

    params_mesh = shard_tree_with_named_spec(params_cpu, param_pspecs, mesh)
    ema_params_mesh = shard_tree_with_named_spec(params_ema_cpu, param_pspecs, mesh)
    opt_mesh = shard_tree_with_named_spec(opt_cpu, opt_pspecs, mesh)
    text_encoder_params_mesh = shard_tree_with_named_spec(text_encoder_params, text_encoder_pspecs, mesh)

    rng, rng_loop = jax.random.split(rng, 2)
    rng_loop = np.asarray(rng_loop)
    train_rng = shard_tree_with_named_spec(rng_loop, PartitionSpec(), mesh)
    ckpt_writer = None

    write_note(f"First step compilations...\n{chrono.note}")

    with mesh:
        for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
            metric.step_start(step)
            batch = multihost_utils.host_local_array_to_global_array(batch, mesh, batch_spec)
            with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
                with chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
                    params_mesh, ema_params_mesh, opt_mesh, train_rng, loss_value, measurements = update_fn(
                        params_mesh, ema_params_mesh, text_encoder_params_mesh, vae_params,
                        opt_mesh, batch, train_rng)
            if (utils.itstime(step, get_steps("log_training"), total_steps, host=0)
                    or chrono.warmup and jax.process_index() == 0):
                for i, sched_fn_cpu in enumerate(sched_fns_cpu):
                    metric.measure(f"global_lr_schedule{i if i else ''}", config.lr*sched_fn_cpu(step - 1))
                l = metric.measure("training_loss", float(loss_value))
                for name, value in measurements.items():
                    metric.measure(name, float(value))
                chrono.tick(step)
                if not np.isfinite(l):
                    raise RuntimeError(f"The losses became nan or inf somewhere within steps "
                                        f"[{step - get_steps('log_training')}, {step}]")
            should_save_ckpt_all_hosts = (
                config.save_ckpt and save_ckpt_path and
                (utils.itstime(step, get_steps("ckpt", None), total_steps, host=None) or
                    utils.itstime(step, get_steps("keep_ckpt", None), total_steps, host=None))
            )
            should_run_autogen_all_hosts = (
                config.save_ckpt and save_ckpt_path and
                utils.itstime(step, get_steps("keep_ckpt", None), total_steps, host=None, first=False)
            )
            # Checkpoint saving
            if should_save_ckpt_all_hosts:
                chrono.pause(wait_for=(params_mesh, ema_params_mesh, opt_mesh))
                if jax.process_index() == 0:
                    checkpointing_timeout(ckpt_writer, config.ckpt_timeout)
                params_cpu = materialize_tree_for_checkpoint(params_mesh)
                params_ema_cpu = materialize_tree_for_checkpoint(ema_params_mesh)
                opt_cpu = materialize_tree_for_checkpoint(opt_mesh)

                if jax.process_index() == 0:
                    # Check whether we want to keep a copy of the current checkpoint.
                    copy_step = None
                    if utils.itstime(step, get_steps("keep_ckpt", None), total_steps):
                        copy_step = step

                    ckpt = {"params": params_cpu, 'params_ema': params_ema_cpu,
                            "opt": opt_cpu, "chrono": chrono.save()}
                    ckpt_writer = pool.apply_async(
                        save_checkpoint, (ckpt, save_ckpt_path, copy_step))
                chrono.resume()
            if auto_generator.enabled and should_run_autogen_all_hosts:
                chrono.pause(wait_for=(params_mesh, ema_params_mesh, opt_mesh))
                try:
                    auto_generator.run_after_checkpoint(step=step, ckpt_writer=ckpt_writer)
                finally:
                    chrono.resume()

            metric.step_end()
            if has_wandb and jax.process_index()==0:
                if config.wandb.log_wandb:
                    wandb.log(metric.step_metrics)
    write_note(f"Done!\n{chrono.note}")
    pool.close()
    pool.join()
    metric.close()
    sync()

if __name__ == "__main__":
    app.run(main)
