"""Dataset utilities for EuroSAT RGB tiles."""

from __future__ import annotations

from pathlib import Path
import random

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


def _attach_snr(
    images: tf.Tensor,
    training: bool,
    snr_db: float,
    snr_db_min: float,
    snr_db_max: float,
):
    batch_size = tf.shape(images)[0]
    if training:
        snr = tf.random.uniform(
            shape=(batch_size, 1),
            minval=snr_db_min,
            maxval=snr_db_max,
            dtype=tf.float32,
        )
    else:
        snr = tf.fill((batch_size, 1), tf.cast(snr_db, tf.float32))
    return {"image": images, "snr_db": snr}, images


def _preprocess_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.image.resize(image, (image_size, image_size), method="bilinear")
    return tf.cast(image, tf.float32) / 255.0


def _class_names_from_local_dir(local_eurosat_dir: str) -> list[str]:
    return sorted([p.name for p in Path(local_eurosat_dir).iterdir() if p.is_dir()])


def _list_local_image_files(local_eurosat_dir: str) -> list[str]:
    class_patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
    files: list[str] = []
    for pattern in class_patterns:
        files.extend(str(p) for p in Path(local_eurosat_dir).glob(f"*/*{pattern[1:]}"))
    return sorted(files)


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
    snr_db: float,
    snr_db_min: float,
    snr_db_max: float,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
):
    files = _list_local_image_files(local_eurosat_dir)

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
        ds = ds.batch(batch_size)
        ds = ds.map(
            lambda images, targets: _attach_snr(
                images,
                training=training,
                snr_db=snr_db,
                snr_db_min=snr_db_min,
                snr_db_max=snr_db_max,
            ),
            num_parallel_calls=autotune,
        )
        return ds.prefetch(autotune)

    return prep(train_paths, True), prep(val_paths, False), prep(test_paths, False)


def _build_local_labeled_test_dataset(
    local_eurosat_dir: str,
    image_size: int,
    batch_size: int,
    train_fraction: float,
    val_fraction: float,
    split_seed: int,
    limit: int | None = None,
):
    files = _list_local_image_files(local_eurosat_dir)
    if not files:
        raise ValueError(
            f"No image files found under '{local_eurosat_dir}'. Expected class subfolders with image files."
        )

    class_names = _class_names_from_local_dir(local_eurosat_dir)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    sorted_files = sorted(files)
    labels = [class_to_idx[Path(path).parent.name] for path in sorted_files]
    all_paths = tf.constant(sorted_files)
    all_labels = tf.constant(labels, dtype=tf.int32)
    shuffled_indices = tf.random.experimental.stateless_shuffle(tf.range(len(sorted_files)), seed=[split_seed, 0])
    shuffled = tf.gather(all_paths, shuffled_indices)
    shuffled_labels = tf.gather(all_labels, shuffled_indices)

    n_total = len(files)
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * val_fraction)
    test_paths = shuffled[n_train + n_val :]
    test_labels = shuffled_labels[n_train + n_val :]
    if limit is not None:
        test_paths = test_paths[:limit]
        test_labels = test_labels[:limit]

    autotune = tf.data.AUTOTUNE

    def decode_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        return _preprocess_image(image, image_size), label

    ds = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    ds = ds.map(decode_image, num_parallel_calls=autotune)
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds, class_names


def sample_random_images(
    image_size: int,
    num_images: int,
    data_dir: str | None = None,
    local_eurosat_dir: str | None = None,
    seed: int = 42,
) -> tf.Tensor:
    """Sample random images from the full EuroSAT dataset and preprocess to [0, 1]."""
    if num_images <= 0:
        raise ValueError("--num-images must be > 0.")

    local_path = _find_local_eurosat(local_eurosat_dir)
    if local_path:
        print(f"Sampling random images from local EuroSAT_RGB at: {local_path}")
        files = _list_local_image_files(local_path)
        if not files:
            raise ValueError(
                f"No image files found under '{local_path}'. Expected class subfolders with image files."
            )

        rng = random.Random(seed)
        if num_images <= len(files):
            selected = rng.sample(files, k=num_images)
        else:
            selected = [rng.choice(files) for _ in range(num_images)]

        images = []
        for path in selected:
            image = tf.io.read_file(path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            images.append(_preprocess_image(image, image_size))
        return tf.stack(images, axis=0)

    print("Local EuroSAT_RGB dataset not found; sampling from tensorflow_datasets eurosat/rgb.")
    ds = tfds.load(name="eurosat/rgb", split="train", as_supervised=True, data_dir=data_dir)
    ds = ds.shuffle(buffer_size=27000, seed=seed, reshuffle_each_iteration=False).take(num_images)

    images = []
    for image, _ in ds:
        images.append(_preprocess_image(image, image_size))

    if not images:
        raise ValueError("Could not sample any images from eurosat/rgb.")
    return tf.stack(images, axis=0)


def build_datasets(
    image_size: int,
    batch_size: int,
    snr_db: float = 10.0,
    train_snr_db_min: float = 10.0,
    train_snr_db_max: float = 10.0,
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
            snr_db=snr_db,
            snr_db_min=train_snr_db_min,
            snr_db_max=train_snr_db_max,
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
        ds = ds.batch(batch_size)
        ds = ds.map(
            lambda images, targets: _attach_snr(
                images,
                training=training,
                snr_db=snr_db,
                snr_db_min=train_snr_db_min,
                snr_db_max=train_snr_db_max,
            ),
            num_parallel_calls=autotune,
        )
        return ds.prefetch(autotune)

    return prep(train_ds, True), prep(val_ds, False), prep(test_ds, False)


def build_labeled_test_dataset(
    image_size: int,
    batch_size: int,
    test_split: str = "train[90%:]",
    data_dir: str | None = None,
    local_eurosat_dir: str | None = None,
    local_train_fraction: float = 0.8,
    local_val_fraction: float = 0.1,
    split_seed: int = 42,
    limit: int | None = None,
):
    """Build labeled test dataset for downstream classification evaluation."""
    local_path = _find_local_eurosat(local_eurosat_dir)
    if local_path:
        print(f"Using local EuroSAT_RGB dataset at: {local_path}")
        return _build_local_labeled_test_dataset(
            local_eurosat_dir=local_path,
            image_size=image_size,
            batch_size=batch_size,
            train_fraction=local_train_fraction,
            val_fraction=local_val_fraction,
            split_seed=split_seed,
            limit=limit,
        )

    print("Local EuroSAT_RGB dataset not found; falling back to tensorflow_datasets eurosat/rgb.")
    ds, info = tfds.load(
        name="eurosat/rgb",
        split=test_split,
        as_supervised=True,
        with_info=True,
        data_dir=data_dir,
    )
    class_names = list(info.features["label"].names)
    autotune = tf.data.AUTOTUNE
    ds = ds.map(lambda x, y: (_preprocess_image(x, image_size), tf.cast(y, tf.int32)), num_parallel_calls=autotune)
    if limit is not None:
        ds = ds.take(limit)
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds, class_names
