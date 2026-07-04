import argparse
import os
import sys
INFERENCE_DIR = os.path.abspath(os.path.dirname(__file__))
PROMPT_DIR = os.path.join(INFERENCE_DIR, "prompts")
PROJECT_ROOT = os.path.abspath(os.path.join(INFERENCE_DIR, ".."))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import jax
if not jax.distributed.is_initialized():
    try:
        jax.distributed.initialize()
    except Exception as e:
        pass

from absl import flags, logging
from absl.flags.argparse_flags import ArgumentParser
import numpy as np
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm
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
from diffusion.inference_pipeline import FlaxInferencePipeline
from text_encoder.text_encoder import TextEncoder
from vae.vae import VAE_CONFIGS, load_vae

from flax.jax_utils import replicate
from flax.training.common_utils import shard
from utils.common import load_checkpoint
from itertools import islice
import tensorflow as tf
from ml_collections import config_flags
import json
import gc

if "config" not in flags.FLAGS:
    config_flags.DEFINE_config_file(
        "config", None, "Training configuration.", lock_config=True)
if not flags.FLAGS.is_parsed():
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


def chunk_list(data, n):
    it = iter(data)
    return iter(lambda: list(islice(it, n)), [])

def sample(
    text,
    masks,
    flax_pipe,
    flax_all_params,
    args,
    model_dtype,
):
    iter_key = jax.random.PRNGKey(args.global_seed)
    key_dist = jax.random.split(iter_key, jax.local_device_count())
    guidance_scale = jnp.full((jax.local_device_count(),), args.cfg_scale, dtype=model_dtype)
    cfg_rescale = jnp.full((jax.local_device_count(),), args.cfg_rescale, dtype=model_dtype)

    samples = flax_pipe(
        flax_all_params,
        text,
        masks,
        key=key_dist,
        guidance_scale=guidance_scale,
        num_inference_steps=args.num_sampling_steps,
        cfg_rescale=cfg_rescale,
    )['images']

    samples = (samples * 255).round().astype("uint8")
    return samples

def pad_to_multiple(x: list[tuple], multiple: int):
    """Pads leading axis of x up to the next multiple of `multiple`."""
    return x + x[:(-len(x)) % multiple]

"""
Helper for T5 Gemma tokenizer
"""
from typing import Sequence
from gemma.gm import data

def _is_str_array(x) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    return np.dtype(x.dtype).type in {np.object_, np.str_}

def _normalize_prompt(prompt: str | Sequence[str]) -> list[str]:
    """Normalize the inputs."""
    if _is_str_array(prompt):  # Supports batched input array
        assert isinstance(prompt, np.ndarray)
        prompt = prompt.tolist()

    return [prompt] if isinstance(prompt, str) else list(prompt)

def _tokenize_prompts(tokenizer, encoder_type: str, captions):
    if encoder_type in {"T5Gemma", "T5Gemma_2B_no_IT", "T5Gemma_9B_2B", "T5Gemma_9B_2B_no_IT"}:
        prompt = _normalize_prompt(list(captions))
        tokens = [tokenizer.tokenizer.encode(p)[:tokenizer.max_input_length] for p in prompt]
        temp = data.pad(tokens, max_length=tokenizer.max_input_length)
        ids = np.array(temp)
        masks = (ids != 0)  # PAD_ID is 0
    else:
        temp = tokenizer(captions, return_tensors="np")
        ids = temp["input_ids"]
        masks = temp["attention_mask"] if temp.get("attention_mask") is not None else np.ones_like(temp["input_ids"])
    return ids, masks

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

def build_dit_model(config, args, latent_size, text_embed_dim, text_num_tokens, model_dtype):
    config_cls, model_cls, presets = _BACKBONE_MODELS[config.backbone]
    preset = presets[config.model_size]
    model_kwargs = dict(config.model_kwargs)
    model_kwargs.setdefault("in_channels", VAE_CONFIGS[config.vae_type]["vae_channels"])
    return model_cls(
        config_cls(
            input_size=latent_size,
            image_resolution=args.image_size,
            patch_size=config.patch_size,
            text_embed_dim=text_embed_dim,
            text_num_tokens=text_num_tokens,
            drop_text_prob=0.1,
            use_qknorm=config.use_qknorm,
            use_swiglu=config.use_swiglu,
            use_rmsnorm=config.use_rmsnorm,
            use_grad_ckpt=config.use_grad_ckpt,
            **preset,
            **model_kwargs,
        ),
        dtype=model_dtype,
    )


def _prompt_path(filename):
    return os.path.join(PROMPT_DIR, filename)


def get_caption_list(name):
    if name == "geneval":
        with open(_prompt_path("geneval.jsonl")) as fp:
            metadatas = [json.loads(line) for line in fp]
        caption_list = [metadata["prompt"] for metadata in metadatas]
    elif name == "geneval_simple_rewrite":
        with open(_prompt_path("geneval_simple_rewrite.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "geneval_complex_rewrite":
        with open(_prompt_path("geneval_complex_rewrite.jsonl")) as fp:
            metadatas = [json.loads(line) for line in fp]
        caption_list = [metadata["prompt"] for metadata in metadatas]
    elif name == "geneval_repeated4":
        with open(_prompt_path("geneval_repeated4.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "geneval_repeated8":
        with open(_prompt_path("geneval_repeated8.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "geneval_repeated12":
        with open(_prompt_path("geneval_repeated12.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "geneval_repeated16":
        with open(_prompt_path("geneval_repeated16.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "geneval_repeated20":
        with open(_prompt_path("geneval_repeated20.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "geneval_repeated24":
        with open(_prompt_path("geneval_repeated24.txt"), "r") as fp:
            caption_list = fp.readlines()
        caption_list = [caption.strip() for caption in caption_list]
    elif name == "dpg":
        with open(_prompt_path("dpg.json"), "r") as f:
            caption_list = json.load(f)
        caption_list = [item[1].strip() for item in caption_list]
    elif name == "dpg_simple_rewrite":
        with open(_prompt_path("dpg_simple_rewrite.json"), "r") as f:
            caption_list = json.load(f)
        caption_list = [item[1].strip() for item in caption_list]
    elif name == "dpg_complex_rewrite":
        with open(_prompt_path("dpg_complex_rewrite.json"), "r") as f:
            caption_list = json.load(f)
        caption_list = [item[1].strip() for item in caption_list]
    elif name == "prism":
        with open(_prompt_path("prism.json"), "r") as f:
            caption_list = json.load(f)
    elif name == "prism_simple_rewrite":
        with open(_prompt_path("prism_simple_rewrite.json"), "r") as f:
            caption_list = json.load(f)
    elif name == "prism_complex_rewrite":
        with open(_prompt_path("prism_complex_rewrite.json"), "r") as f:
            caption_list = json.load(f)
    elif name == "CVTG-2K":
        with open(_prompt_path("CVTG-2K.json"), "r") as f:
            caption_list = json.load(f)
        caption_list = [item[1].strip() for item in caption_list]
    elif name == "CVTG-2K_simple_rewrite":
        with open(_prompt_path("CVTG-2K_simple_rewrite.json"), "r") as f:
            caption_list = json.load(f)
        caption_list = [item[1].strip() for item in caption_list]
    elif name == "CVTG-2K_complex_rewrite":
        with open(_prompt_path("CVTG-2K_complex_rewrite.json"), "r") as f:
            caption_list = json.load(f)
        caption_list = [item[1].strip() for item in caption_list]
    elif name == "longtext":
        caption_list = []
        with open(_prompt_path("longtext.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                caption_list.append(obj["prompt"])
    elif name == "longtext_simple_rewrite":
        caption_list = []
        with open(_prompt_path("longtext_simple_rewrite.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                caption_list.append(obj["prompt"])
    elif name == "longtext_complex_rewrite":
        caption_list = []
        with open(_prompt_path("longtext_complex_rewrite.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                caption_list.append(obj["prompt"])
    else:
        raise ValueError(f"No caption list found for {name}")
    return caption_list

def run_with_config(args, config):
    tf.config.experimental.set_visible_devices([], "GPU")
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    vae_config = VAE_CONFIGS[config.vae_type]
    latent_size = args.image_size // vae_config["vae_compression_factor"]

    model_dtype = jnp.float32
    text_encoder_dtype = get_weight_dtype(config.text_encoder_precision)
    text_encoder_bundle = TextEncoder(
        config,
        config.text_encoder_type,
        config.token_len,
        weight_dtype=text_encoder_dtype,
        dit_checkpoint_path=args.ckpt,
    )
    tokenizer, jax_text_encoder, text_encoder_params = text_encoder_bundle.tokenizer, text_encoder_bundle.text_encoder, text_encoder_bundle.text_encoder_params
    text_encoder_params = cast_tree_to_dtype(text_encoder_params, text_encoder_dtype)
    is_multi_text_encoder = isinstance(jax_text_encoder, (list, tuple))
    jax_dit_model = build_dit_model(
        config,
        args,
        latent_size,
        text_encoder_bundle.hidden_dim,
        text_encoder_bundle.text_token_len,
        model_dtype,
    )
    init_key = jax.random.PRNGKey(0)
    x = jnp.zeros((2, vae_config["vae_channels"], latent_size, latent_size), model_dtype)
    t = jnp.linspace(0, 1000, 2, dtype=model_dtype)
    if is_multi_text_encoder:
        assert len(text_encoder_bundle.text_token_len) == len(text_encoder_bundle.hidden_dim)
        y = [
            jnp.zeros((2, cur_len, cur_dim), dtype=model_dtype)
            for cur_len, cur_dim in zip(text_encoder_bundle.text_token_len, text_encoder_bundle.hidden_dim)
        ]
        total_tokens = int(sum(text_encoder_bundle.text_token_len))
        mask = jnp.ones((2, total_tokens), dtype=model_dtype)
    else:
        y = jnp.zeros((2, text_encoder_bundle.text_token_len, text_encoder_bundle.hidden_dim), dtype=model_dtype)
        mask = jnp.ones((2, text_encoder_bundle.text_token_len), dtype=model_dtype)
    dit_params = jax_dit_model.init(init_key, x, t, y, mask=mask)

    jax_vae_model, vae_params = load_vae(config)
    flax_pipe = FlaxInferencePipeline(
        jax_dit_model,
        jax_vae_model,
        jax_text_encoder,
        config=config,
        dtype=model_dtype,
    )

    process_index = jax.process_index()
    process_count = jax.process_count()
    caption_list = get_caption_list(args.prompt_type)
    if ("geneval" in args.prompt_type) or ("dpg" in args.prompt_type) or ("longtext" in args.prompt_type):
        caption_list = [([it, idx_for_same_prompt], caption) for it, caption in enumerate(caption_list) for idx_for_same_prompt in range(4)]
    else:
        caption_list = [(it, caption) for it, caption in enumerate(caption_list)]
    caption_list = pad_to_multiple(caption_list, jax.device_count())
    captions_per_worker = len(caption_list) // process_count
    start_idx = process_index * captions_per_worker
    end_idx = start_idx + captions_per_worker if process_index < process_count - 1 else len(caption_list)
    worker_captions = caption_list[start_idx:end_idx]
    import math
    total_batches = math.ceil(len(worker_captions) / args.per_proc_batch_size)
    cfg_scales = args.cfg_scale if isinstance(args.cfg_scale, (list, tuple)) else [args.cfg_scale]
    cfg_rescales = args.cfg_rescale if isinstance(args.cfg_rescale, (list, tuple)) else [args.cfg_rescale]
    iter_values = args.checkpoint_iters if isinstance(args.checkpoint_iters, (list, tuple)) else [args.checkpoint_iters]
    base_sample_dir = args.sample_dir
    ckpt_template = args.ckpt
    for iter_value in iter_values:
        padded_iter = f"{int(iter_value):09d}"
        ckpt_path = f"{ckpt_template}-{padded_iter}"
        args.ckpt = ckpt_path

        logging.info(f'Loading from JAX checkpoint: {ckpt_path}')
        loaded = load_checkpoint(None, ckpt_path)
        dit_params = cast_tree_to_dtype(loaded["params_ema"], model_dtype)
        del loaded

        if getattr(args, "sync_after_sampling", True):
            jax.experimental.multihost_utils.sync_global_devices("hf_weights_ready")

        # Reuse one replicated params tree for all CFG settings of this checkpoint.
        flax_all_params = dict(vae=vae_params, text_encoder=text_encoder_params, transformer=dit_params)
        flax_all_params = replicate(flax_all_params)

        try:
            for cfg_scale in cfg_scales:
                for cfg_rescale in cfg_rescales:
                    args.cfg_scale = cfg_scale
                    args.cfg_rescale = cfg_rescale
                    current_sample_dir = f"{base_sample_dir}_{padded_iter}_cfg{float(cfg_scale):.1f}_rescale{float(cfg_rescale):.1f}"
                    args.sample_dir = current_sample_dir
                    sample_folder_dir = current_sample_dir
                    os.makedirs(sample_folder_dir, exist_ok=True)
                    logging.info(f"Saving .png samples at {sample_folder_dir}")
                    batches = chunk_list(worker_captions, args.per_proc_batch_size)
                    progress_desc = f"Sampling cfg={float(cfg_scale):.1f} rescale={float(cfg_rescale):.1f} iter={padded_iter}"
                    for step, batch in enumerate(tqdm(batches, total=total_batches, desc=progress_desc), start=1):
                        its, captions = zip(*batch)
                        if is_multi_text_encoder:
                            ids = []
                            masks = []
                            for enc_type, enc_tokenizer in zip(text_encoder_bundle.text_encoder_type, tokenizer):
                                enc_ids, enc_masks = _tokenize_prompts(enc_tokenizer, enc_type, captions)
                                ids.append(enc_ids)
                                masks.append(enc_masks)
                        else:
                            ids, masks = _tokenize_prompts(tokenizer, text_encoder_bundle.text_encoder_type, captions)
                        if is_multi_text_encoder:
                            ids = [shard(enc_ids) for enc_ids in ids]
                            masks = [shard(enc_masks) for enc_masks in masks]
                        else:
                            ids = shard(ids)
                            masks = shard(masks)

                        images = sample(
                            ids,
                            masks,
                            flax_pipe,
                            flax_all_params,
                            args,
                            model_dtype,
                        )
                        if ("geneval" in args.prompt_type) or ("dpg" in args.prompt_type):
                            images = images.reshape(-1, args.image_size, args.image_size, 3)
                            for idx, img_arr in zip(its, images):
                                idx, idx_for_same_prompt = idx
                                save_folder_same_prompt = os.path.join(sample_folder_dir, f"{idx:0>5}", "samples")
                                os.makedirs(save_folder_same_prompt, exist_ok=True)
                                img = Image.fromarray(np.squeeze(img_arr))
                                img.save(os.path.join(save_folder_same_prompt, f"{idx_for_same_prompt:05}.png"))
                        elif ("longtext" in args.prompt_type):
                            images = images.reshape(-1, args.image_size, args.image_size, 3)
                            for idx, img_arr in zip(its, images):
                                idx, idx_for_same_prompt = idx
                                img = Image.fromarray(np.squeeze(img_arr))
                                img.save(os.path.join(sample_folder_dir, f"{idx:0>4}_{idx_for_same_prompt}.png"))
                        elif ("prism" in args.prompt_type) or ("CVTG-2K" in args.prompt_type):
                            images = images.reshape(-1, args.image_size, args.image_size, 3)
                            for idx, img_arr in zip(its, images):
                                img = Image.fromarray(np.squeeze(img_arr))
                                img.save(os.path.join(sample_folder_dir, f"{idx:05d}.png"))
                        else:
                            raise ValueError(f"No prompt type found for {args.prompt_type}")
                        del images
        finally:
            del flax_all_params
            del dit_params
            gc.collect()
    if getattr(args, "sync_after_sampling", True):
        jax.experimental.multihost_utils.sync_global_devices("samples_done")


def main(args):
    """
    Run sampling using config loaded from absl flags.
    This is to handle the case where this generate.py (instead of training/main.py) is the main entry point.
    """
    return run_with_config(args, flags.FLAGS.config)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="sample")
    parser.add_argument("--per-proc-batch-size", type=int, default=256)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--cfg-scale",  type=float, nargs="+", default=[12.0])
    parser.add_argument("--cfg-rescale", type=float, nargs="+", default=[1.0])
    parser.add_argument("--checkpoint-iters", type=int, nargs="+", required=True)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=99)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--prompt-type", type=str, default="dpg")
    parser.add_argument("--check-gcs-region", type=str2bool, default=True)
    args = parser.parse_args()

    if args.check_gcs_region:
        import requests
        from google.cloud import storage
        client = storage.Client()
        METADATA_URL = "http://metadata.google.internal/computeMetadata/v1/instance/zone"
        HEADERS = {"Metadata-Flavor": "Google"}
        tpu_zone_path = requests.get(METADATA_URL, headers=HEADERS, timeout=20).text
        tpu_zone = tpu_zone_path.split("/")[-1]
        tpu_region = "-".join(tpu_zone.split("-")[:-1])
        bucket_name = args.ckpt.removeprefix("gs://").split("/")[0]
        bucket = client.get_bucket(bucket_name)
        bucket_region = bucket.location.lower()
        if (tpu_region is None) or (len(tpu_region) < 8) or (bucket_region is None) or (len(bucket_region) < 8):
            print(f"One of them is empty: TPU region {tpu_region}, bucket region {bucket_region}", file=sys.stderr, flush=True)
            import sys; sys.exit(2)
        elif bucket_region != tpu_region:
            print(f"TPU region {tpu_region} doesn't match bucket region {bucket_region}", file=sys.stderr, flush=True)
            import sys; sys.exit(2)
        else:
            print(f"TPU region {tpu_region} matches bucket region {bucket_region}, proceed", file=sys.stderr, flush=True)

    main(args)
