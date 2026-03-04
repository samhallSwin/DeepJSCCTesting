#!/usr/bin/env python3
"""CLI for training/evaluating DeepJSCC on EuroSAT RGB 64x64 tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from deepjscc.channels import CHANNEL_CHOICES
from deepjscc.data import build_datasets
from deepjscc.model import DeepJSCC, PSNRMetric


def build_model(args):
    model = DeepJSCC(
        image_size=args.image_size,
        channel_uses=args.channel_uses,
        latent_channels=args.latent_channels,
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


def train(args):
    train_ds, val_ds, _ = build_datasets(
        image_size=args.image_size,
        batch_size=args.batch_size,
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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

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
    if args.weights:
        model.load_weights(args.weights)

    metrics = model.evaluate(test_ds, return_dict=True)
    print(json.dumps(metrics, indent=2))


def parser():
    p = argparse.ArgumentParser(description="DeepJSCC for EuroSAT RGB")
    sub = p.add_subparsers(dest="command", required=True)

    def add_shared(sp):
        sp.add_argument("--image-size", type=int, default=64)
        sp.add_argument("--batch-size", type=int, default=64)
        sp.add_argument("--channel-uses", type=int, default=256)
        sp.add_argument("--latent-channels", type=int, default=128)
        sp.add_argument("--channel-type", choices=CHANNEL_CHOICES, default="awgn")
        sp.add_argument("--snr-db", type=float, default=10.0)
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

    p_train = sub.add_parser("train", help="Train DeepJSCC model")
    add_shared(p_train)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--output-dir", type=str, default="artifacts/deepjscc")
    p_train.set_defaults(func=train)

    p_eval = sub.add_parser("evaluate", help="Evaluate DeepJSCC model")
    add_shared(p_eval)
    p_eval.add_argument("--weights", type=str, default=None)
    p_eval.set_defaults(func=evaluate)

    return p


def main():
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
