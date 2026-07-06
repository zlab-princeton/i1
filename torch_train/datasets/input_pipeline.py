import numpy as np
import tensorflow as tf
import torch

import datasets.build_transforms as pp_builder
import datasets.tfds as ds_tfds


def _add_host_options(data):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    options.threading.max_intra_op_parallelism = 1
    return data.with_options(options)


def build_single_source_train_dataset(data, preprocess_fn, shuffle_buffer_size, num_parallel_calls=100):
    data = _add_host_options(data)
    data = data.repeat(None)
    data = data.shuffle(shuffle_buffer_size) if shuffle_buffer_size else data
    data = data.map(preprocess_fn, num_parallel_calls=num_parallel_calls)
    return data.prefetch(2)


def build_training_dataset(input_config, process_index: int = 0, process_count: int = 1):
    if not isinstance(input_config["data"], (list, tuple)):
        raise TypeError("input_config['data'] must be a list of (dataset_cfg, weight) pairs.")

    datasets, weights, ntraining_examples = [], [], 0
    for dataset_cfg, weight in input_config["data"]:
        train_data = ds_tfds.DataSource(
            split=dataset_cfg["split"],
            data_dir=dataset_cfg["data_dir"],
            process_index=process_index,
            process_count=process_count,
        )
        ntraining_examples += train_data.total_examples
        dataset = build_single_source_train_dataset(
            data=train_data.get_tfdata(),
            preprocess_fn=pp_builder.get_preprocess_fn(input_config["preprocess"]),
            shuffle_buffer_size=int(input_config["shuffle_buffer_size"] * weight),
        )
        datasets.append(dataset)
        weights.append(float(weight))
    weight_sum = sum(weights)
    weights = [x / weight_sum for x in weights]
    train_ds = tf.data.Dataset.sample_from_datasets(datasets, weights, stop_on_empty_dataset=True)
    train_ds = train_ds.batch(input_config["batch_size"] // process_count, drop_remainder=True)
    return train_ds, ntraining_examples


def tokenize_batch(batch, tokenizer, token_len):
    labels = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in batch["labels"].tolist()]
    tok = tokenizer(
        labels,
        max_length=token_len,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return {
        "image": torch.from_numpy(np.asarray(batch["image"])),
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
    }


def start_input_iterator(train_ds, tokenizer, token_len):
    return (tokenize_batch(batch, tokenizer, token_len) for batch in train_ds.as_numpy_iterator())
