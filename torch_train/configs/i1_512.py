from configs.i1_256 import get_config as _base_config


def get_config():
    config = _base_config()
    config.image_size = 512
    config.total_steps = 2_500_000

    path_and_count = [
        ('/path/to/512_resolution_imagenet22k/tfrecord', 1.0),
        ('/path/to/512_resolution_rendered_text/tfrecord', 1.0),
        ('/path/to/512_resolution_fluxreason/tfrecord', 1.0),
        ('/path/to/512_resolution_textatlas/tfrecord', 1.0),
        ('/path/to/512_resolution_pexels/tfrecord', 1.0),
        ('/path/to/512_resolution_gptedit/tfrecord', 1.0),
        ('/path/to/512_resolution_midjourneyv6/tfrecord', 1.0),
        ('/path/to/512_resolution_redcaps/tfrecord', 1.0),
        ('/path/to/512_resolution_places/tfrecord', 1.0),
        ('/path/to/512_resolution_megalith10m/tfrecord', 1.0),
    ]
    sum_count = sum(item[1] for item in path_and_count)
    path_and_count = [(item[0], float(item[1]) / float(sum_count)) for item in path_and_count]
    config.input.data = [(dict(split="train", data_dir=data_dir), w) for data_dir, w in path_and_count]

    config.resume = ""  # set to your 256-resolution checkpoint.pt (or use --resume)
    return config
