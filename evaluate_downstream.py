#!/usr/bin/env python3
"""Evaluate downstream EuroSAT land-cover classification on original and reconstructed images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from deepjscc.channels import CHANNEL_CHOICES
from deepjscc.data import build_labeled_test_dataset
from deepjscc.downstream_classifier import EuroSATClassifier
from deepjscc.model import MODEL_VARIANTS, DeepJSCC
from deepjscc.sionna_link import simulate_real_ldpc_link_sionna
from traditional_baseline import (
    _channel_capacity_bpcu,
    compress_image_to_payload,
    decode_payload_to_image,
    max_source_bits_for_real_ldpc,
    simulate_real_ldpc_link,
)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    idx = y_true * num_classes + y_pred
    return np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _classification_summary(name: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    cm = _confusion_matrix(y_true, y_pred, len(class_names))
    accuracy = float(np.trace(cm) / max(1, cm.sum()))
    per_class_accuracy = []
    f1_scores = []
    for idx, class_name in enumerate(class_names):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - tp)
        fp = float(cm[:, idx].sum() - tp)
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        f1_scores.append(f1)
        per_class_accuracy.append(
            {
                "class_name": class_name,
                "accuracy": tp / max(1.0, tp + fn),
                "support": int(cm[idx, :].sum()),
            }
        )
    return {
        "name": name,
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_scores)),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }


def _build_deepjscc(args) -> DeepJSCC:
    model = DeepJSCC(
        image_size=args.image_size,
        channel_uses=args.channel_uses,
        latent_channels=args.latent_channels,
        model_variant=args.model_variant,
        channel_type=args.channel_type,
        snr_db=args.snr_db,
        rician_k=args.rician_k,
    )
    _ = model(
        {
            "image": tf.zeros((1, args.image_size, args.image_size, 3), dtype=tf.float32),
            "snr_db": tf.fill((1, 1), tf.cast(args.snr_db, tf.float32)),
        },
        training=False,
    )
    model.load_weights(args.deepjscc_weights)
    return model


def _reconstruct_deepjscc(model: DeepJSCC, images: tf.Tensor, snr_db: float) -> tf.Tensor:
    recon = model(
        {
            "image": images,
            "snr_db": tf.fill((tf.shape(images)[0], 1), tf.cast(snr_db, tf.float32)),
        },
        training=False,
    )
    return tf.cast(tf.clip_by_value(recon, 0.0, 1.0), tf.float32)


def _reconstruct_traditional_image(image: tf.Tensor, args, capacity_bpcu: float, effective_source_bits_budget: int, seed: int):
    payload, info = compress_image_to_payload(
        image01=image,
        codec=args.codec,
        target_bits=effective_source_bits_budget,
        bpg_q_min=args.bpg_q_min,
        bpg_q_max=args.bpg_q_max,
    )
    if args.link_model == "real-ldpc" and info["source_bits"] > effective_source_bits_budget:
        return tf.zeros_like(image), False

    if args.link_model == "ideal":
        coded_rate_bpcu = (info["source_bits"] / args.ldpc_rate) / args.channel_uses
        if coded_rate_bpcu <= capacity_bpcu:
            return decode_payload_to_image(payload, info["codec_used"]), True
        return tf.zeros_like(image), False

    if args.ldpc_backend == "sionna":
        success, rx_payload, _ = simulate_real_ldpc_link_sionna(
            payload=payload,
            channel_uses=args.channel_uses,
            ldpc_rate=args.ldpc_rate,
            mod_order=args.mod_order,
            snr_db=args.snr_db,
            codeword_length=args.ldpc_codeword_length,
            bp_iters=args.ldpc_iters,
            seed=seed,
        )
    else:
        success, rx_payload, _ = simulate_real_ldpc_link(
            payload=payload,
            channel_uses=args.channel_uses,
            ldpc_rate=args.ldpc_rate,
            mod_order=args.mod_order,
            snr_db=args.snr_db,
            codeword_length=args.ldpc_codeword_length,
            row_weight=args.ldpc_row_weight,
            bp_iters=args.ldpc_iters,
            seed=seed,
        )
    if not success:
        return tf.zeros_like(image), False
    try:
        return decode_payload_to_image(rx_payload, info["codec_used"]), True
    except Exception:
        return tf.zeros_like(image), False


def _reconstruct_traditional_batch(images: tf.Tensor, args, capacity_bpcu: float, effective_source_bits_budget: int, seed_offset: int):
    recon = []
    successes = []
    for i in range(images.shape[0]):
        rec, ok = _reconstruct_traditional_image(
            image=images[i],
            args=args,
            capacity_bpcu=capacity_bpcu,
            effective_source_bits_budget=effective_source_bits_budget,
            seed=seed_offset + i,
        )
        recon.append(rec)
        successes.append(ok)
    return tf.stack(recon, axis=0), successes


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Downstream EuroSAT land-cover classification evaluation")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-images", type=int, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--test-split", type=str, default="train[90%:]")
    p.add_argument("--local-eurosat-dir", type=str, default=None)
    p.add_argument("--local-train-fraction", type=float, default=0.8)
    p.add_argument("--local-val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--classifier-model-id", type=str, default="cm93/resnet18-eurosat")
    p.add_argument("--classifier-device", type=str, default="cpu")

    p.add_argument("--deepjscc-weights", type=str, required=True)
    p.add_argument("--model-variant", choices=tuple(MODEL_VARIANTS.keys()), default="base")
    p.add_argument("--latent-channels", type=int, default=128)

    p.add_argument("--channel-type", choices=CHANNEL_CHOICES, default="awgn")
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--rician-k", type=float, default=5.0)
    p.add_argument("--channel-uses", type=int, default=256)

    p.add_argument("--ldpc-rate", type=float, default=0.5)
    p.add_argument("--mod-order", type=int, default=4)
    p.add_argument("--codec", choices=["bpg", "jpeg"], default="bpg")
    p.add_argument("--bpg-q-min", type=int, default=0)
    p.add_argument("--bpg-q-max", type=int, default=51)
    p.add_argument("--capacity-mc-samples", type=int, default=20000)
    p.add_argument("--link-model", choices=["ideal", "real-ldpc"], default="ideal")
    p.add_argument("--ldpc-backend", choices=["custom", "sionna"], default="custom")
    p.add_argument("--ldpc-codeword-length", type=int, default=512)
    p.add_argument("--ldpc-row-weight", type=int, default=3)
    p.add_argument("--ldpc-iters", type=int, default=30)

    p.add_argument("--output-path", type=str, default="artifacts/downstream_eval_summary.json")
    return p


def main():
    args = parser().parse_args()
    if args.link_model == "real-ldpc" and args.channel_type != "awgn":
        raise ValueError("--link-model real-ldpc currently supports --channel-type awgn only.")
    if args.link_model == "real-ldpc" and args.mod_order not in (2, 4):
        raise ValueError("--link-model real-ldpc currently supports --mod-order 2 or 4.")

    ds, dataset_class_names = build_labeled_test_dataset(
        image_size=args.image_size,
        batch_size=args.batch_size,
        test_split=args.test_split,
        data_dir=args.data_dir,
        local_eurosat_dir=args.local_eurosat_dir,
        local_train_fraction=args.local_train_fraction,
        local_val_fraction=args.local_val_fraction,
        split_seed=args.split_seed,
        limit=args.num_images,
    )
    classifier = EuroSATClassifier(model_id=args.classifier_model_id, device=args.classifier_device)
    label_map = classifier.build_label_mapping(dataset_class_names)
    deepjscc_model = _build_deepjscc(args)

    bits_per_channel_use = math.log2(args.mod_order)
    source_bits_budget = int(args.channel_uses * bits_per_channel_use * args.ldpc_rate)
    effective_source_bits_budget = source_bits_budget
    ldpc_fit_bits = None
    if args.link_model == "real-ldpc":
        ldpc_fit_bits, _ = max_source_bits_for_real_ldpc(
            channel_uses=args.channel_uses,
            mod_order=args.mod_order,
            ldpc_codeword_length=args.ldpc_codeword_length,
            ldpc_rate=args.ldpc_rate,
        )
        effective_source_bits_budget = min(source_bits_budget, ldpc_fit_bits)

    capacity_bpcu = _channel_capacity_bpcu(
        channel_type=args.channel_type,
        snr_db=args.snr_db,
        rician_k=args.rician_k,
        mc_samples=args.capacity_mc_samples,
        seed=args.seed,
    )

    y_true_all = []
    orig_pred_all = []
    deep_pred_all = []
    trad_pred_all = []
    traditional_successes = 0
    traditional_total = 0

    for batch_index, (images, labels) in enumerate(ds):
        labels_np = labels.numpy().astype(np.int64)
        mapped_labels = label_map[labels_np]
        y_true_all.append(mapped_labels)

        orig_pred_all.append(classifier.predict(images))
        deep_recon = _reconstruct_deepjscc(deepjscc_model, images, snr_db=args.snr_db)
        deep_pred_all.append(classifier.predict(deep_recon))

        trad_recon, batch_successes = _reconstruct_traditional_batch(
            images=images,
            args=args,
            capacity_bpcu=capacity_bpcu,
            effective_source_bits_budget=effective_source_bits_budget,
            seed_offset=args.seed + batch_index * args.batch_size,
        )
        trad_pred_all.append(classifier.predict(trad_recon))
        traditional_successes += int(sum(batch_successes))
        traditional_total += len(batch_successes)

    y_true = np.concatenate(y_true_all, axis=0)
    orig_pred = np.concatenate(orig_pred_all, axis=0)
    deep_pred = np.concatenate(deep_pred_all, axis=0)
    trad_pred = np.concatenate(trad_pred_all, axis=0)

    summary = {
        "classifier": {
            "model_id": args.classifier_model_id,
            "device": args.classifier_device,
            "label_names": classifier.label_names,
        },
        "channel": {
            "channel_type": args.channel_type,
            "snr_db": args.snr_db,
            "rician_k": args.rician_k,
            "channel_uses": args.channel_uses,
        },
        "traditional": {
            "codec": args.codec,
            "link_model": args.link_model,
            "ldpc_backend": args.ldpc_backend,
            "success_rate": traditional_successes / max(1, traditional_total),
        },
        "original": _classification_summary("original", y_true, orig_pred, classifier.label_names),
        "deepjscc": _classification_summary("deepjscc", y_true, deep_pred, classifier.label_names),
        "baseline": _classification_summary("baseline", y_true, trad_pred, classifier.label_names),
    }
    summary["deepjscc"]["accuracy_drop_vs_original"] = (
        summary["original"]["accuracy"] - summary["deepjscc"]["accuracy"]
    )
    summary["baseline"]["accuracy_drop_vs_original"] = (
        summary["original"]["accuracy"] - summary["baseline"]["accuracy"]
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
