"""Dataset utilities for EuroSAT RGB tiles."""

from __future__ import annotations

from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


LOCAL_EUROSAT_CANDIDATES = (
    "../datasets/EuroSAT_RGB",
    "../../datasets/EuroSAT_RGB",
    "datasets/EuroSAT_RGB",
)


def _preprocess(image, _label, image_size: int):
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image, image


def _find_local_eurosat(local_eurosat_dir: str | None) -> str | None:
    if local_eurosat_dir:
        candidate = Path(local_eurosat_dir).expanduser().resolve()
        return str(candidate) if candidate.is_dir() else None

    for candidate in LOCAL_EUROSAT_CANDIDATES:
        path = Path(candidate).expanduser().resolve()
        if path.is_dir():
            return str(path)
    return None


def _build_local_datasets(
    local_eurosat_dir: str,
    image_size: int,
    batch_size: int,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
):
    class_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files: list[str] = []
    for pattern in class_patterns:
        files.extend(str(p) for p in Path(local_eurosat_dir).glob(f"*/*{pattern[1:]}"))

    if not files:
        raise ValueError(
            f"No image files found under '{local_eurosat_dir}'. Expected class subfolders with image files."
        )

    files = sorted(files)
    n_total = len(files)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)

    if n_train <= 0 or n_train + n_val >= n_total:
        raise ValueError(
            "Invalid local split sizes. Adjust --local-train-fraction and --local-val-fraction."
        )

    all_paths = tf.constant(files)
    shuffled = tf.random.experimental.stateless_shuffle(all_paths, seed=[split_seed, 0])

    train_paths = shuffled[:n_train]
    val_paths = shuffled[n_train : n_train + n_val]
    test_paths = shuffled[n_train + n_val :]

    autotune = tf.data.AUTOTUNE

    def decode_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return image, tf.constant(0, dtype=tf.int32)

    def prep(path_ds, training):
        ds = tf.data.Dataset.from_tensor_slices(path_ds)
        if training:
            ds = ds.shuffle(min(4096, n_train), seed=split_seed, reshuffle_each_iteration=True)
        ds = ds.map(decode_image, num_parallel_calls=autotune)
        ds = ds.map(lambda x, y: _preprocess(x, y, image_size), num_parallel_calls=autotune)
        return ds.batch(batch_size).prefetch(autotune)

    return prep(train_paths, True), prep(val_paths, False), prep(test_paths, False)


def build_datasets(
    image_size: int,
    batch_size: int,
    train_split: str = "train[:80%]",
    val_split: str = "train[80%:90%]",
    test_split: str = "train[90%:]",
    data_dir: str | None = None,
    local_eurosat_dir: str | None = None,
    local_train_fraction: float = 0.8,
    local_val_fraction: float = 0.1,
    split_seed: int = 42,
):
    """Build train/val/test datasets, preferring a local EuroSAT_RGB directory when present."""
    local_path = _find_local_eurosat(local_eurosat_dir)
    if local_path:
        print(f"Using local EuroSAT_RGB dataset at: {local_path}")
        return _build_local_datasets(
            local_eurosat_dir=local_path,
            image_size=image_size,
            batch_size=batch_size,
            train_fraction=local_train_fraction,
            val_fraction=local_val_fraction,
            split_seed=split_seed,
        )

    print("Local EuroSAT_RGB dataset not found; falling back to tensorflow_datasets eurosat/rgb.")
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
