from configs.i1_256 import get_config as _base_config


def get_config():
    config = _base_config()
    config.image_size = 1024
    config.total_steps = 2_800_000

    path_and_count = [
        ('/path/to/1024_resolution_fluxreason/tfrecord', 1.0),
        ('/path/to/1024_resolution_textatlas/tfrecord', 1.0),
        ('/path/to/1024_resolution_gptedit/tfrecord', 1.0),
        ('/path/to/1024_resolution_midjourneyv6/tfrecord', 1.0),
        ('/path/to/1024_resolution_redcaps/tfrecord', 1.0),
    ]
    sum_count = sum(item[1] for item in path_and_count)
    path_and_count = [(item[0], float(item[1]) / float(sum_count)) for item in path_and_count]
    config.input.data = [(dict(split="train", data_dir=data_dir), w) for data_dir, w in path_and_count]
    config.input.batch_size = 128

    config.transport.train_timestep_shift = 0.3
    config.resume = ""  # set to your 512-resolution checkpoint.pt (or use --resume)
    return config
