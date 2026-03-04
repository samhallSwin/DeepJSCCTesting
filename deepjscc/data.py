"""Dataset utilities for EuroSAT RGB tiles."""

from __future__ import annotations

import tensorflow as tf
import tensorflow_datasets as tfds


def _preprocess(image, _label, image_size: int):
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image, image


def build_datasets(
    image_size: int,
    batch_size: int,
    train_split: str = "train[:80%]",
    val_split: str = "train[80%:90%]",
    test_split: str = "train[90%:]",
    data_dir: str | None = None,
):
    """Build train/val/test datasets from tfds EuroSAT RGB."""
    ds_kwargs = dict(name="eurosat/rgb", as_supervised=True, data_dir=data_dir)

    train_ds = tfds.load(split=train_split, **ds_kwargs)
    val_ds = tfds.load(split=val_split, **ds_kwargs)
    test_ds = tfds.load(split=test_split, **ds_kwargs)

    autotune = tf.data.AUTOTUNE

    def prep(ds, training):
        ds = ds.map(lambda x, y: _preprocess(x, y, image_size), num_parallel_calls=autotune)
        if training:
            ds = ds.shuffle(4096, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size).prefetch(autotune)
        return ds

    return prep(train_ds, True), prep(val_ds, False), prep(test_ds, False)
