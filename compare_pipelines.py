#!/usr/bin/env python3
"""Compare DeepJSCC and traditional baseline on the same random image set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from deepjscc.channels import CHANNEL_CHOICES
from deepjscc.data import sample_random_images
from deepjscc.model import MODEL_VARIANTS, DeepJSCC
from traditional_baseline import (
    _channel_capacity_bpcu,
    _mae,
    _psnr,
    _tensor_to_u8,
    compress_image_to_payload,
    decode_payload_to_image,
    max_source_bits_for_real_ldpc,
    simulate_real_ldpc_link,
)
from deepjscc.sionna_link import simulate_real_ldpc_link_sionna


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare DeepJSCC vs traditional BPG/JPEG+LDPC baseline on same random images"
    )
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--num-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--local-eurosat-dir", type=str, default=None)

    # Shared channel settings.
    p.add_argument("--channel-type", choices=CHANNEL_CHOICES, default="awgn")
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--rician-k", type=float, default=5.0)
    p.add_argument("--channel-uses", type=int, default=256)

    # DeepJSCC settings.
    p.add_argument("--deepjscc-weights", type=str, required=True)
    p.add_argument("--model-variant", choices=tuple(MODEL_VARIANTS.keys()), default="base")
    p.add_argument("--latent-channels", type=int, default=128)

    # Traditional settings.
    p.add_argument("--ldpc-rate", type=float, default=0.5)
    p.add_argument(
        "--mod-order", type=int, default=4, help="QAM order (2=BPSK, 4=QPSK, 16=16-QAM, ...)"
    )
    p.add_argument("--codec", choices=["bpg", "jpeg"], default="bpg")
    p.add_argument("--bpg-q-min", type=int, default=0)
    p.add_argument("--bpg-q-max", type=int, default=51)
    p.add_argument("--capacity-mc-samples", type=int, default=20000)
    p.add_argument("--link-model", choices=["ideal", "real-ldpc"], default="ideal")
    p.add_argument("--ldpc-backend", choices=["custom", "sionna"], default="custom")
    p.add_argument("--ldpc-codeword-length", type=int, default=512)
    p.add_argument("--ldpc-row-weight", type=int, default=3)
    p.add_argument("--ldpc-iters", type=int, default=30)

    p.add_argument("--output-dir", type=str, default="artifacts/pipeline_comparison")
    return p


def build_deepjscc(args) -> DeepJSCC:
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


def main():
    args = parser().parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be > 0.")
    if args.ldpc_rate <= 0.0 or args.ldpc_rate > 1.0:
        raise ValueError("--ldpc-rate must be in (0, 1].")
    if args.mod_order <= 1 or (args.mod_order & (args.mod_order - 1)) != 0:
        raise ValueError("--mod-order must be a power of two >= 2.")
    if args.channel_uses <= 0:
        raise ValueError("--channel-uses must be > 0.")
    if args.link_model == "real-ldpc" and args.channel_type != "awgn":
        raise ValueError("--link-model real-ldpc currently supports --channel-type awgn only.")
    if args.link_model == "real-ldpc" and args.mod_order not in (2, 4):
        raise ValueError("--link-model real-ldpc currently supports --mod-order 2 or 4.")
    if args.link_model == "real-ldpc" and args.ldpc_backend == "sionna":
        try:
            import sionna  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Sionna backend requested but `sionna` is not installed in this environment. "
                "Install with: pip install -r requirements.txt -r requirements-sionna.txt"
            ) from exc

    out = Path(args.output_dir)
    originals_dir = out / "originals"
    deepjscc_dir = out / "deepjscc"
    traditional_dir = out / "traditional"
    comparisons_dir = out / "comparisons"
    for d in [originals_dir, deepjscc_dir, traditional_dir, comparisons_dir]:
        d.mkdir(parents=True, exist_ok=True)

    images = sample_random_images(
        image_size=args.image_size,
        num_images=args.num_images,
        data_dir=args.data_dir,
        local_eurosat_dir=args.local_eurosat_dir,
        seed=args.seed,
    )

    deepjscc_model = build_deepjscc(args)
    deepjscc_recon = tf.clip_by_value(
        tf.cast(
            deepjscc_model(
                {
                    "image": images,
                    "snr_db": tf.fill((tf.shape(images)[0], 1), tf.cast(args.snr_db, tf.float32)),
                },
                training=False,
            ),
            tf.float32,
        ),
        0.0,
        1.0,
    )

    bits_per_channel_use = math.log2(args.mod_order)
    source_bits_budget = int(args.channel_uses * bits_per_channel_use * args.ldpc_rate)
    ldpc_fit_bits = None
    ldpc_fit_details = None
    effective_source_bits_budget = source_bits_budget
    if args.link_model == "real-ldpc":
        ldpc_fit_bits, ldpc_fit_details = max_source_bits_for_real_ldpc(
            channel_uses=args.channel_uses,
            mod_order=args.mod_order,
            ldpc_codeword_length=args.ldpc_codeword_length,
            ldpc_rate=args.ldpc_rate,
        )
        effective_source_bits_budget = min(source_bits_budget, ldpc_fit_bits)

    image_pixels = args.image_size * args.image_size
    raw_bits_per_image = image_pixels * 3 * 8
    target_source_bpp = effective_source_bits_budget / image_pixels
    target_compression_ratio = raw_bits_per_image / max(1, effective_source_bits_budget)
    channel_uses_per_pixel = args.channel_uses / image_pixels
    capacity_bpcu = _channel_capacity_bpcu(
        channel_type=args.channel_type,
        snr_db=args.snr_db,
        rician_k=args.rician_k,
        mc_samples=args.capacity_mc_samples,
        seed=args.seed,
    )

    per_image = []
    deepjscc_psnr, deepjscc_mae = [], []
    traditional_psnr, traditional_mae = [], []
    traditional_outages = 0

    for i in range(args.num_images):
        original = images[i]
        deep = deepjscc_recon[i]

        payload, info = compress_image_to_payload(
            image01=original,
            codec=args.codec,
            target_bits=effective_source_bits_budget,
            bpg_q_min=args.bpg_q_min,
            bpg_q_max=args.bpg_q_max,
        )
        if args.link_model == "real-ldpc" and info["source_bits"] > effective_source_bits_budget:
            trad_success = False
            coded_rate_bpcu = (info["source_bits"] / args.ldpc_rate) / args.channel_uses
            trad = tf.zeros_like(original)
        elif args.link_model == "ideal":
            coded_rate_bpcu = (info["source_bits"] / args.ldpc_rate) / args.channel_uses
            trad_success = coded_rate_bpcu <= capacity_bpcu
            if trad_success:
                trad = decode_payload_to_image(payload, info["codec_used"])
            else:
                trad = tf.zeros_like(original)
        else:
            if args.ldpc_backend == "sionna":
                trad_success, rx_payload, link_info = simulate_real_ldpc_link_sionna(
                    payload=payload,
                    channel_uses=args.channel_uses,
                    ldpc_rate=args.ldpc_rate,
                    mod_order=args.mod_order,
                    snr_db=args.snr_db,
                    codeword_length=args.ldpc_codeword_length,
                    bp_iters=args.ldpc_iters,
                    seed=args.seed + i,
                )
            else:
                trad_success, rx_payload, link_info = simulate_real_ldpc_link(
                    payload=payload,
                    channel_uses=args.channel_uses,
                    ldpc_rate=args.ldpc_rate,
                    mod_order=args.mod_order,
                    snr_db=args.snr_db,
                    codeword_length=args.ldpc_codeword_length,
                    row_weight=args.ldpc_row_weight,
                    bp_iters=args.ldpc_iters,
                    seed=args.seed + i,
                )
            coded_rate_bpcu = link_info.get("coded_bits", 0) / max(1, args.channel_uses)
            if trad_success:
                try:
                    trad = decode_payload_to_image(rx_payload, info["codec_used"])
                except Exception:
                    trad_success = False
                    trad = tf.zeros_like(original)
            else:
                trad = tf.zeros_like(original)
        if not trad_success:
            traditional_outages += 1

        d_psnr = _psnr(original, deep)
        d_mae = _mae(original, deep)
        t_psnr = _psnr(original, trad)
        t_mae = _mae(original, trad)

        deepjscc_psnr.append(d_psnr)
        deepjscc_mae.append(d_mae)
        traditional_psnr.append(t_psnr)
        traditional_mae.append(t_mae)

        original_u8 = _tensor_to_u8(original)
        deep_u8 = _tensor_to_u8(deep)
        trad_u8 = _tensor_to_u8(trad)
        comp = tf.concat([original_u8, deep_u8, trad_u8], axis=1)

        stem = f"img_{i:03d}"
        tf.io.write_file(str(originals_dir / f"{stem}.jpg"), tf.io.encode_jpeg(original_u8))
        tf.io.write_file(str(deepjscc_dir / f"{stem}.jpg"), tf.io.encode_jpeg(deep_u8))
        tf.io.write_file(str(traditional_dir / f"{stem}.jpg"), tf.io.encode_jpeg(trad_u8))
        tf.io.write_file(str(comparisons_dir / f"{stem}.jpg"), tf.io.encode_jpeg(comp))

        per_image.append(
            {
                "index": i,
                "deepjscc_psnr": d_psnr,
                "deepjscc_mae": d_mae,
                "traditional_psnr": t_psnr,
                "traditional_mae": t_mae,
                "traditional_success": bool(trad_success),
                "traditional_codec_used": info["codec_used"],
                "traditional_quality": info["quality"],
                "traditional_source_bits": info["source_bits"],
                "traditional_source_bpp": info["source_bits"] / image_pixels,
                "traditional_compression_ratio": raw_bits_per_image / max(1, info["source_bits"]),
                "traditional_coded_rate_bpcu": coded_rate_bpcu,
            }
        )

    mean_traditional_source_bits = (
        float(np.mean([x["traditional_source_bits"] for x in per_image])) if per_image else None
    )

    summary = {
        "num_images": args.num_images,
        "image_size": args.image_size,
        "image_pixels": image_pixels,
        "raw_bits_per_image": raw_bits_per_image,
        "channel": {
            "channel_type": args.channel_type,
            "snr_db": args.snr_db,
            "rician_k": args.rician_k,
            "channel_uses": args.channel_uses,
            "channel_uses_per_pixel": channel_uses_per_pixel,
        },
        "compression_view": {
            "source_bits_budget": source_bits_budget,
            "effective_source_bits_budget": effective_source_bits_budget,
            "real_ldpc_fit_source_bits_max": ldpc_fit_bits,
            "real_ldpc_fit_details": ldpc_fit_details,
            "target_source_bpp": target_source_bpp,
            "target_compression_ratio": target_compression_ratio,
            "equivalent_mapping": "bpp_eq = channel_uses * log2(mod_order) * ldpc_rate / (H*W)",
        },
        "deepjscc": {
            "weights": args.deepjscc_weights,
            "model_variant": args.model_variant,
            "latent_channels": args.latent_channels,
            "mean_psnr": float(np.mean(deepjscc_psnr)),
            "mean_mae": float(np.mean(deepjscc_mae)),
        },
        "traditional": {
            "codec_requested": args.codec,
            "link_model": args.link_model,
            "ldpc_backend": args.ldpc_backend,
            "ldpc_rate": args.ldpc_rate,
            "mod_order": args.mod_order,
            "bits_per_channel_use": bits_per_channel_use,
            "source_bits_budget": source_bits_budget,
            "effective_source_bits_budget": effective_source_bits_budget,
            "target_source_bpp": target_source_bpp,
            "target_compression_ratio": target_compression_ratio,
            "mean_source_bits": mean_traditional_source_bits,
            "mean_source_bpp": (
                mean_traditional_source_bits / image_pixels if mean_traditional_source_bits is not None else None
            ),
            "mean_compression_ratio": (
                raw_bits_per_image / mean_traditional_source_bits if mean_traditional_source_bits else None
            ),
            "capacity_bpcu_estimate": capacity_bpcu,
            "outage_count": traditional_outages,
            "outage_rate": traditional_outages / max(1, args.num_images),
            "mean_psnr": float(np.mean(traditional_psnr)),
            "mean_mae": float(np.mean(traditional_mae)),
            "ldpc_impl": {
                "codeword_length": args.ldpc_codeword_length,
                "row_weight": args.ldpc_row_weight,
                "bp_iters": args.ldpc_iters,
            },
        },
        "output_dir": str(out.resolve()),
    }

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(out / "per_image.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
