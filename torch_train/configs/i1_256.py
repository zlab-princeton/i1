from utils.config import ConfigDict


def get_config():
    config = ConfigDict()

    # Run.
    config.seed = 0
    config.total_steps = 2_000_000
    config.resume = ""

    # Model.
    config.image_size = 256
    config.vae_type = "flux2"
    config.in_channels = 32  # flux2 vae channels
    config.text_encoder_type = "T5Gemma"
    config.text_encoder_precision = "bf16"
    config.token_len = None
    config.backbone = "dual_stream"
    config.model_size = "DiT-XL_2016"
    config.patch_size = 2
    config.model_kwargs = ConfigDict(dict(
        rope_axes_dims=None,
        rope_axes_lens=None,
        rope_theta=10000.0,
        use_long_skip=True,
        text_encoder_adapter_type="transformer",
        text_encoder_adapter_num_blocks=2,
        use_image_connector=False,
        use_adaln=False,
        repeat_text_emb=False,
        position_embedding="sinusoidal_and_rope",
        use_sandwich_norm=True,
        use_separate_norms=False,
    ))
    config.use_qknorm = True
    config.use_swiglu = True
    config.use_rmsnorm = True
    config.use_grad_ckpt = False
    config.amp = True  # bf16 mixed precision

    config.transport = ConfigDict(dict(
        prediction="velocity",
        use_lognorm=True,
        lognorm_mu=0.0,
        lognorm_sigma=1.0,
        train_timestep_shift=0.0,
        cfg_interval_start=0,
    ))

    # Parallelism.
    config.tensor_parallel_size = 1
    config.fsdp_axis_size = 1

    # Data.
    path_and_count = [
        ('/path/to/yfcc/tfrecord', 1.0),
        ('/path/to/imagenet22k/tfrecord', 1.0),
        ('/path/to/rendered_text/tfrecord', 1.0),
        ('/path/to/fluxreason/tfrecord', 1.0),
        ('/path/to/textatlas/tfrecord', 1.0),
        ('/path/to/pexels/tfrecord', 1.0),
        ('/path/to/gptedit/tfrecord', 1.0),
        ('/path/to/midjourneyv6/tfrecord', 1.0),
        ('/path/to/redcaps/tfrecord', 1.0),
        ('/path/to/places/tfrecord', 1.0),
        ('/path/to/megalith10m/tfrecord', 1.0),
    ]
    sum_count = sum(item[1] for item in path_and_count)
    path_and_count = [(item[0], float(item[1]) / float(sum_count)) for item in path_and_count]

    config.input = ConfigDict()
    config.input.data = [(dict(split="train", data_dir=data_dir), w) for data_dir, w in path_and_count]
    config.input.batch_size = 512
    config.input.shuffle_buffer_size = 10_000
    config.input.preprocess = (
        "|decode_png()"
        '|sample_caption(key="caption")'
        "|value_range(-1, 1)"
        '|copy("caption", "labels")'
        '|keep("image", "labels")'
    )

    # Runtime.
    config.grad_accum_steps = 1
    config.log_training_steps = 100
    config.ckpt_steps = 5000
    config.keep_ckpt_steps = 50000
    config.save_ckpt = True

    # Optimizer and schedule.
    config.grad_clip_norm = 1.0
    config.b1 = 0.9
    config.b2 = 0.95
    config.adam_eps = 1e-8
    config.mu_dtype = "bfloat16"
    config.lr = 0.0001
    config.freeze_patterns = ["pos_embed"]

    # EMA.
    config.use_ema = True
    config.ema_decay_rate = 0.9999

    # Logging.
    config.wandb = ConfigDict(dict(
        log_wandb=False,
        project="i1",
        experiment="experiment_name",
    ))

    return config
