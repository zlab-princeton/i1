import tensorflow_datasets as tfds


class DataSource:
    def __init__(self, split, data_dir, process_index: int = 0, process_count: int = 1):
        self.builder = tfds.builder_from_directory(data_dir)
        self.split = split
        process_splits = tfds.even_splits(split, process_count)
        self.process_split = process_splits[process_index]
        self.skip_decoders = {
            f: tfds.decode.SkipDecoding()
            for f in ("image",)
            if f in self.builder.info.features
        }

    def get_tfdata(self):
        return self.builder.as_dataset(
            split=self.process_split,
            shuffle_files=True,
            read_config=tfds.ReadConfig(
                skip_prefetch=True,
                try_autocache=False,
                add_tfds_id=True,
            ),
            decoders=self.skip_decoders,
        )

    @property
    def total_examples(self):
        return self.builder.info.splits[self.split].num_examples
