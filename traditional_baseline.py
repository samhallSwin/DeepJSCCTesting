#!/usr/bin/env python3
"""Traditional digital baseline: BPG/JPEG + LDPC-rate-constrained link model."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

from deepjscc.channels import CHANNEL_CHOICES
from deepjscc.data import build_datasets


def _tensor_to_u8(image01: tf.Tensor) -> tf.Tensor:
    return tf.image.convert_image_dtype(tf.clip_by_value(image01, 0.0, 1.0), tf.uint8)


def _psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    return float(tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0)).numpy())


def _mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    return float(tf.reduce_mean(tf.abs(y_true - y_pred)).numpy())


def _channel_capacity_bpcu(channel_type: str, snr_db: float, rician_k: float, mc_samples: int, seed: int) -> float:
    """Capacity in bits per complex channel use under ideal coding assumptions."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    channel_type = channel_type.lower()

    if channel_type == "none":
        return float("inf")
    if channel_type == "awgn":
        return float(math.log2(1.0 + snr_linear))

    rng = np.random.default_rng(seed)
    if channel_type == "rayleigh":
        power = rng.exponential(scale=1.0, size=mc_samples)
    elif channel_type == "rician":
        los = math.sqrt(rician_k / (rician_k + 1.0))
        scatter_sigma = math.sqrt(1.0 / (2.0 * (rician_k + 1.0)))
        h_real = los + rng.normal(loc=0.0, scale=scatter_sigma, size=mc_samples)
        h_imag = rng.normal(loc=0.0, scale=scatter_sigma, size=mc_samples)
        power = h_real**2 + h_imag**2
    else:
        raise ValueError(f"Unsupported channel_type '{channel_type}'.")

    return float(np.mean(np.log2(1.0 + power * snr_linear)))


def _compress_jpeg_to_budget_bits(image_u8: tf.Tensor, target_bits: int) -> tuple[bytes, int, int]:
    """Return best-quality JPEG bytes that satisfy the target bits if possible."""
    best_fit = None
    for quality in range(100, 0, -1):
        encoded = tf.io.encode_jpeg(image_u8, quality=quality).numpy()
        bits = len(encoded) * 8
        if bits <= target_bits:
            best_fit = (encoded, bits, quality)
            break
    if best_fit is not None:
        return best_fit

    # Could not fit within budget even at minimum quality; return lowest quality.
    encoded = tf.io.encode_jpeg(image_u8, quality=1).numpy()
    return encoded, len(encoded) * 8, 1


def _compress_bpg_to_budget_bits(
    image_u8: tf.Tensor,
    target_bits: int,
    bpgenc_path: str,
    q_min: int,
    q_max: int,
) -> tuple[bytes, int, int]:
    """
    Compress image to BPG at highest quality within target bits.
    BPG quality q: 0=best quality (largest size), 51=lowest quality (smallest size).
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_png = td_path / "in.png"
        out_bpg = td_path / "out.bpg"
        tf.io.write_file(str(in_png), tf.io.encode_png(image_u8))

        best_fit = None
        for q in range(q_min, q_max + 1):
            cmd = [bpgenc_path, "-q", str(q), "-o", str(out_bpg), str(in_png)]
            subprocess.run(cmd, check=True, capture_output=True)
            payload = out_bpg.read_bytes()
            bits = len(payload) * 8
            if bits <= target_bits:
                best_fit = (payload, bits, q)
                break

        if best_fit is not None:
            return best_fit

        # Could not fit budget; return lowest quality (largest q).
        cmd = [bpgenc_path, "-q", str(q_max), "-o", str(out_bpg), str(in_png)]
        subprocess.run(cmd, check=True, capture_output=True)
        payload = out_bpg.read_bytes()
        return payload, len(payload) * 8, q_max


def _decode_jpeg_bytes(payload: bytes) -> tf.Tensor:
    image = tf.image.decode_jpeg(payload, channels=3)
    return tf.cast(image, tf.float32) / 255.0


def _decode_bpg_bytes(payload: bytes, bpgdec_path: str) -> tf.Tensor:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_bpg = td_path / "in.bpg"
        out_png = td_path / "out.png"
        in_bpg.write_bytes(payload)
        cmd = [bpgdec_path, "-o", str(out_png), str(in_bpg)]
        subprocess.run(cmd, check=True, capture_output=True)
        image = tf.image.decode_png(tf.io.read_file(str(out_png)), channels=3)
        return tf.cast(image, tf.float32) / 255.0


def _compress_and_reconstruct(
    image01: tf.Tensor,
    codec: str,
    target_bits: int,
    bpg_q_min: int,
    bpg_q_max: int,
) -> tuple[tf.Tensor, dict]:
    image_u8 = _tensor_to_u8(image01)
    bpgenc = shutil.which("bpgenc")
    bpgdec = shutil.which("bpgdec")
    use_bpg = codec == "bpg" and bpgenc and bpgdec

    if use_bpg:
        payload, source_bits, quality = _compress_bpg_to_budget_bits(
            image_u8=image_u8,
            target_bits=target_bits,
            bpgenc_path=bpgenc,
            q_min=bpg_q_min,
            q_max=bpg_q_max,
        )
        recon = _decode_bpg_bytes(payload, bpgdec_path=bpgdec)
        codec_used = "bpg"
    else:
        payload, source_bits, quality = _compress_jpeg_to_budget_bits(image_u8, target_bits=target_bits)
        recon = _decode_jpeg_bytes(payload)
        codec_used = "jpeg"

    info = {
        "codec_used": codec_used,
        "source_bits": int(source_bits),
        "quality": int(quality),
    }
    return recon, info


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traditional baseline: BPG/JPEG + LDPC-rate constrained link")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--channel-type", choices=CHANNEL_CHOICES, default="awgn")
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--rician-k", type=float, default=5.0)
    p.add_argument("--channel-uses", type=int, default=256)
    p.add_argument("--ldpc-rate", type=float, default=0.5)
    p.add_argument("--mod-order", type=int, default=4, help="QAM order (2=BPSK, 4=QPSK, 16=16-QAM, ...)")
    p.add_argument("--codec", choices=["bpg", "jpeg"], default="bpg")
    p.add_argument("--bpg-q-min", type=int, default=0)
    p.add_argument("--bpg-q-max", type=int, default=51)
    p.add_argument("--num-images", type=int, default=512)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--train-split", type=str, default="train[:80%]")
    p.add_argument("--val-split", type=str, default="train[80%:90%]")
    p.add_argument("--test-split", type=str, default="train[90%:]")
    p.add_argument("--local-eurosat-dir", type=str, default=None)
    p.add_argument("--local-train-fraction", type=float, default=0.8)
    p.add_argument("--local-val-fraction", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--capacity-mc-samples", type=int, default=20000)
    p.add_argument("--save-images", type=int, default=8)
    p.add_argument("--output-dir", type=str, default="artifacts/traditional_baseline")
    return p


def main():
    args = parser().parse_args()
    if args.ldpc_rate <= 0.0 or args.ldpc_rate > 1.0:
        raise ValueError("--ldpc-rate must be in (0, 1].")
    if args.mod_order <= 1 or (args.mod_order & (args.mod_order - 1)) != 0:
        raise ValueError("--mod-order must be a power of two >= 2.")
    if args.channel_uses <= 0:
        raise ValueError("--channel-uses must be > 0.")
    if args.num_images <= 0:
        raise ValueError("--num-images must be > 0.")

    output_dir = Path(args.output_dir)
    (output_dir / "originals").mkdir(parents=True, exist_ok=True)
    (output_dir / "reconstructions").mkdir(parents=True, exist_ok=True)
    (output_dir / "comparisons").mkdir(parents=True, exist_ok=True)

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

    bits_per_channel_use = math.log2(args.mod_order)
    source_bits_budget = int(args.channel_uses * bits_per_channel_use * args.ldpc_rate)
    image_pixels = args.image_size * args.image_size
    raw_bits_per_image = image_pixels * 3 * 8
    target_bpp = source_bits_budget / image_pixels
    target_compression_ratio = raw_bits_per_image / max(1, source_bits_budget)
    capacity_bpcu = _channel_capacity_bpcu(
        channel_type=args.channel_type,
        snr_db=args.snr_db,
        rician_k=args.rician_k,
        mc_samples=args.capacity_mc_samples,
        seed=args.split_seed,
    )

    maes: list[float] = []
    psnrs: list[float] = []
    outages = 0
    used_codec_counts = {"bpg": 0, "jpeg": 0}
    per_image = []

    images = test_ds.unbatch().map(lambda x, _y: x).take(args.num_images)
    for i, image in enumerate(images):
        recon_source, info = _compress_and_reconstruct(
            image01=image,
            codec=args.codec,
            target_bits=source_bits_budget,
            bpg_q_min=args.bpg_q_min,
            bpg_q_max=args.bpg_q_max,
        )
        used_codec_counts[info["codec_used"]] += 1

        coded_rate_bpcu = (info["source_bits"] / args.ldpc_rate) / args.channel_uses
        success = coded_rate_bpcu <= capacity_bpcu
        if not success:
            outages += 1
            recon = tf.zeros_like(image)
        else:
            recon = recon_source

        mae = _mae(image, recon)
        psnr = _psnr(image, recon)
        maes.append(mae)
        psnrs.append(psnr)

        if i < args.save_images:
            orig_u8 = _tensor_to_u8(image)
            recon_u8 = _tensor_to_u8(recon)
            side = tf.concat([orig_u8, recon_u8], axis=1)
            stem = f"img_{i:03d}"
            tf.io.write_file(str(output_dir / "originals" / f"{stem}.jpg"), tf.io.encode_jpeg(orig_u8))
            tf.io.write_file(str(output_dir / "reconstructions" / f"{stem}.jpg"), tf.io.encode_jpeg(recon_u8))
            tf.io.write_file(str(output_dir / "comparisons" / f"{stem}.jpg"), tf.io.encode_jpeg(side))

        per_image.append(
            {
                "index": i,
                "success": bool(success),
                "codec_used": info["codec_used"],
                "quality": info["quality"],
                "source_bits": info["source_bits"],
                "source_bpp": info["source_bits"] / image_pixels,
                "compression_ratio": raw_bits_per_image / max(1, info["source_bits"]),
                "coded_rate_bpcu": coded_rate_bpcu,
                "mae": mae,
                "psnr": psnr,
            }
        )

    mean_source_bits = float(np.mean([x["source_bits"] for x in per_image])) if per_image else None

    summary = {
        "num_images": len(per_image),
        "image_size": args.image_size,
        "image_pixels": image_pixels,
        "raw_bits_per_image": raw_bits_per_image,
        "channel_type": args.channel_type,
        "snr_db": args.snr_db,
        "rician_k": args.rician_k,
        "channel_uses": args.channel_uses,
        "ldpc_rate": args.ldpc_rate,
        "mod_order": args.mod_order,
        "bits_per_channel_use": bits_per_channel_use,
        "source_bits_budget": source_bits_budget,
        "target_source_bpp": target_bpp,
        "target_compression_ratio": target_compression_ratio,
        "mean_source_bits": mean_source_bits,
        "mean_source_bpp": (mean_source_bits / image_pixels) if mean_source_bits is not None else None,
        "mean_compression_ratio": (raw_bits_per_image / mean_source_bits) if mean_source_bits else None,
        "capacity_bpcu_estimate": capacity_bpcu,
        "outage_count": outages,
        "outage_rate": outages / max(1, len(per_image)),
        "mean_mae": float(np.mean(maes)) if maes else None,
        "mean_psnr": float(np.mean(psnrs)) if psnrs else None,
        "codec_requested": args.codec,
        "codec_used_counts": used_codec_counts,
        "output_dir": str(output_dir.resolve()),
        "assumptions": {
            "link_model": "Idealized LDPC success when coded spectral efficiency <= estimated channel capacity.",
            "failure_mode": "Outage images reconstructed as zeros.",
            "fallback": "If BPG binaries are unavailable, JPEG is used automatically.",
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
