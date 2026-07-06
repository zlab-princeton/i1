import ml_collections as mlc


def get_config():
    config = mlc.ConfigDict()

    # Run.
    config.seed = 0
    config.total_steps = 2_000_000
    config.resume = ""

    # Model.
    config.image_size = 256
    config.vae_type = "flux2"
    config.text_encoder_type = "T5Gemma"
    config.text_encoder_precision = "bf16"
    config.token_len = None
    config.backbone = "dual_stream"
    config.model_size = 'DiT-XL_2016'
    config.patch_size = 2
    config.model_kwargs = mlc.ConfigDict(dict(
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
    config.shard_text_encoder_params = False

    """
    Reference for these default values in LightningDiT
    use_lognorm: https://github.com/hustvl/LightningDiT/blob/2725fed42a14898744433809949834e26957bcdd/configs/config_details.yaml#L61
    lognorm_mu, lognorm_sigma: https://github.com/hustvl/LightningDiT/blob/2725fed42a14898744433809949834e26957bcdd/transport/transport.py#L155
    """
    config.transport = mlc.ConfigDict(dict(
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
        ('gs://path/to/yfcc/tfrecord', 1),
        ('gs://path/to/imagenet22k/tfrecord', 1),
        ('gs://path/to/rendered_text/tfrecord', 1),
        ('gs://path/to/fluxreason/tfrecord', 1),
        ('gs://path/to/textatlas/tfrecord', 1),
        ('gs://path/to/pexels/tfrecord', 1),
        ('gs://path/to/gptedit/tfrecord', 1),
        ('gs://path/to/midjourneyv6/tfrecord', 1),
        ('gs://path/to/redcaps/tfrecord', 1),
        ('gs://path/to/places/tfrecord', 1),
        ('gs://path/to/megalith10m/tfrecord', 1),
    ]
    sum_count = sum([item[1] for item in path_and_count])
    path_and_count = [(item[0], float(item[1])/float(sum_count)) for item in path_and_count]

    config.input = {}
    config.input.data = [(dict(split='train', data_dir=data_dir), data_weight) for data_dir, data_weight in path_and_count]
    config.input.batch_size = 512
    config.input.shuffle_buffer_size = 10_000 # this is per worker
    config.input.preprocess = (
        '|decode_png()'
        '|sample_caption(key="caption")'
        '|value_range(-1, 1)'
        '|copy("caption", "labels")'
        '|keep("image", "labels")'
    )

    # Runtime.
    config.prefetch_to_device = 4
    config.grad_accum_steps = 1
    config.log_training_steps = 100
    config.ckpt_steps = 5000
    config.keep_ckpt_steps = 50000
    config.save_ckpt = True
    config.ckpt_timeout = 120

    # Optimizer and schedule.
    config.grad_clip_norm = 1.0
    config.optax_name = 'scale_by_adam'
    config.optax = dict(mu_dtype='bfloat16', b1=0.9, b2=0.95)
    config.lr = 0.0001
    config.wd = None
    config.schedule = [
        ('pos_embed', None),
        ('.*', 1.0),
    ]

    # EMA.
    config.use_ema = True
    config.ema_decay_rate = 0.9999

    # Logging.
    config.wandb = dict(
        log_wandb=True,
        wandb_offline=False,
        resume=False,
        project='i1',
        experiment='experiment_name',
    )

    return config
