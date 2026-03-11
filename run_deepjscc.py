#!/usr/bin/env python3
"""CLI for training/evaluating DeepJSCC on EuroSAT RGB 64x64 tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from deepjscc.channels import CHANNEL_CHOICES
from deepjscc.data import build_datasets, sample_random_images
from deepjscc.model import MODEL_VARIANTS, DeepJSCC, PSNRMetric


def configure_runtime(args):
    gpus = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpus:
        raise RuntimeError(
            "No GPU detected by TensorFlow. Install GPU TensorFlow deps and verify CUDA driver/runtime compatibility."
        )

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")


def build_model(args):
    model = DeepJSCC(
        image_size=args.image_size,
        channel_uses=args.channel_uses,
        latent_channels=args.latent_channels,
        model_variant=args.model_variant,
        channel_type=args.channel_type,
        snr_db=args.snr_db,
        rician_k=args.rician_k,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[PSNRMetric(), tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def write_architecture_report(args, model: DeepJSCC, output_dir: Path):
    lines: list[str] = []
    lines.append("DeepJSCC Architecture Report")
    lines.append("")
    lines.append(f"model_variant: {args.model_variant}")
    lines.append(f"image_size: {args.image_size}")
    lines.append(f"channel_uses: {args.channel_uses}")
    lines.append(f"latent_channels: {args.latent_channels}")
    lines.append(f"channel_type: {args.channel_type}")
    lines.append(f"snr_db: {args.snr_db}")
    lines.append(f"train_snr_db_min: {args.train_snr_db_min}")
    lines.append(f"train_snr_db_max: {args.train_snr_db_max}")
    lines.append(f"rician_k: {args.rician_k}")
    lines.append("")
    lines.append(f"variant_spec: {model.variant}")
    lines.append("")
    lines.append("Encoder Summary")
    lines.append("-" * 80)
    model.encoder.summary(print_fn=lines.append)
    lines.append("")
    lines.append("Decoder Summary")
    lines.append("-" * 80)
    model.decoder.summary(print_fn=lines.append)
    lines.append("")
    lines.append("Full Model Summary")
    lines.append("-" * 80)
    model.summary(print_fn=lines.append)
    (output_dir / "architecture.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args):
    train_ds, val_ds, _ = build_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
        snr_db=args.snr_db,
        train_snr_db_min=args.train_snr_db_min,
        train_snr_db_max=args.train_snr_db_max,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        data_dir=args.data_dir,
        local_eurosat_dir=args.local_eurosat_dir,
        local_train_fraction=args.local_train_fraction,
        local_val_fraction=args.local_val_fraction,
        split_seed=args.split_seed,
    )

    model = build_model(args)
    _ = model(
        {
            "image": tf.zeros((1, args.image_size, args.image_size, 3), dtype=tf.float32),
            "snr_db": tf.fill((1, 1), tf.cast(args.snr_db, tf.float32)),
        },
        training=False,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    write_architecture_report(args, model, Path(args.output_dir))

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(Path(args.output_dir) / "best.weights.h5"),
            monitor="val_psnr",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_weights(Path(args.output_dir) / "last.weights.h5")
    with open(Path(args.output_dir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)


def evaluate(args):
    _, _, test_ds = build_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
        snr_db=args.snr_db,
        train_snr_db_min=args.train_snr_db_min,
        train_snr_db_max=args.train_snr_db_max,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        data_dir=args.data_dir,
        local_eurosat_dir=args.local_eurosat_dir,
        local_train_fraction=args.local_train_fraction,
        local_val_fraction=args.local_val_fraction,
        split_seed=args.split_seed,
    )

    model = build_model(args)
    # Subclassed models must be built before loading weights.
    _ = model(
        {
            "image": tf.zeros((1, args.image_size, args.image_size, 3), dtype=tf.float32),
            "snr_db": tf.fill((1, 1), tf.cast(args.snr_db, tf.float32)),
        },
        training=False,
    )
    if args.weights:
        model.load_weights(args.weights)

    metrics = model.evaluate(test_ds, return_dict=True)
    print(json.dumps(metrics, indent=2))


def sample(args):
    images = sample_random_images(
        image_size=args.image_size,
        num_images=args.num_images,
        data_dir=args.data_dir,
        local_eurosat_dir=args.local_eurosat_dir,
        seed=args.seed,
    )

    model = build_model(args)
    _ = model(
        {
            "image": tf.zeros((1, args.image_size, args.image_size, 3), dtype=tf.float32),
            "snr_db": tf.fill((1, 1), tf.cast(args.snr_db, tf.float32)),
        },
        training=False,
    )
    model.load_weights(args.weights)

    recon = model(
        {
            "image": images,
            "snr_db": tf.fill((tf.shape(images)[0], 1), tf.cast(args.snr_db, tf.float32)),
        },
        training=False,
    )
    recon = tf.cast(recon, tf.float32)
    recon = tf.clip_by_value(recon, 0.0, 1.0)

    originals_dir = Path(args.output_dir) / "originals"
    recons_dir = Path(args.output_dir) / "reconstructions"
    comparisons_dir = Path(args.output_dir) / "comparisons"
    originals_dir.mkdir(parents=True, exist_ok=True)
    recons_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    n = images.shape[0]
    for i in range(n):
        orig_u8 = tf.image.convert_image_dtype(tf.clip_by_value(images[i], 0.0, 1.0), tf.uint8)
        recon_u8 = tf.image.convert_image_dtype(recon[i], tf.uint8)
        side_by_side = tf.concat([orig_u8, recon_u8], axis=1)

        stem = f"img_{i:03d}"
        tf.io.write_file(str(originals_dir / f"{stem}.jpg"), tf.io.encode_jpeg(orig_u8))
        tf.io.write_file(str(recons_dir / f"{stem}.jpg"), tf.io.encode_jpeg(recon_u8))
        tf.io.write_file(str(comparisons_dir / f"{stem}.jpg"), tf.io.encode_jpeg(side_by_side))

    manifest = {
        "num_images": int(n),
        "weights": str(args.weights),
        "model_variant": args.model_variant,
        "channel_type": args.channel_type,
        "snr_db": args.snr_db,
        "image_size": args.image_size,
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    with open(Path(args.output_dir) / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


def parser():
    p = argparse.ArgumentParser(description="DeepJSCC for EuroSAT RGB")
    sub = p.add_subparsers(dest="command", required=True)

    def add_shared(sp):
        sp.add_argument("--image-size", type=int, default=64)
        sp.add_argument("--batch-size", type=int, default=64)
        sp.add_argument("--channel-uses", type=int, default=256)
        sp.add_argument("--latent-channels", type=int, default=128)
        sp.add_argument("--model-variant", choices=tuple(MODEL_VARIANTS.keys()), default="base")
        sp.add_argument("--channel-type", choices=CHANNEL_CHOICES, default="awgn")
        sp.add_argument("--snr-db", type=float, default=10.0)
        sp.add_argument("--train-snr-db-min", type=float, default=10.0)
        sp.add_argument("--train-snr-db-max", type=float, default=10.0)
        sp.add_argument("--rician-k", type=float, default=5.0)
        sp.add_argument("--learning-rate", type=float, default=1e-3)
        sp.add_argument("--data-dir", type=str, default=None)
        sp.add_argument("--train-split", type=str, default="train[:80%]")
        sp.add_argument("--val-split", type=str, default="train[80%:90%]")
        sp.add_argument("--test-split", type=str, default="train[90%:]")
        sp.add_argument("--local-eurosat-dir", type=str, default=None)
        sp.add_argument("--local-train-fraction", type=float, default=0.8)
        sp.add_argument("--local-val-fraction", type=float, default=0.1)
        sp.add_argument("--split-seed", type=int, default=42)
        sp.add_argument("--require-gpu", action="store_true")
        sp.add_argument("--mixed-precision", action="store_true")

    p_train = sub.add_parser("train", help="Train DeepJSCC model")
    add_shared(p_train)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--output-dir", type=str, default="artifacts/deepjscc")
    p_train.set_defaults(func=train)

    p_eval = sub.add_parser("evaluate", help="Evaluate DeepJSCC model")
    add_shared(p_eval)
    p_eval.add_argument("--weights", type=str, default=None)
    p_eval.set_defaults(func=evaluate)

    p_sample = sub.add_parser("sample", help="Save random original/reconstruction JPEGs")
    add_shared(p_sample)
    p_sample.add_argument("--weights", type=str, required=True)
    p_sample.add_argument("--num-images", type=int, default=8)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--output-dir", type=str, default="artifacts/deepjscc_samples")
    p_sample.set_defaults(func=sample)

    return p


def main():
    args = parser().parse_args()
    if args.train_snr_db_min > args.train_snr_db_max:
        raise ValueError("--train-snr-db-min must be <= --train-snr-db-max.")
    configure_runtime(args)
    args.func(args)


if __name__ == "__main__":
    main()
