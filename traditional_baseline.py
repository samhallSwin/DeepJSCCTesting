#!/usr/bin/env python3
"""Traditional digital baseline: BPG/JPEG + link model (ideal or real LDPC)."""

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
from deepjscc.ldpc_codec import build_systematic_ldpc, decode_blocks_min_sum, encode_blocks
from deepjscc.sionna_link import simulate_real_ldpc_link_sionna


def _ldpc_k_from_n_rate(n: int, rate: float) -> int:
    k = int(round(n * rate))
    return max(1, min(k, n - 1))


def max_source_bits_for_real_ldpc(
    channel_uses: int,
    mod_order: int,
    ldpc_codeword_length: int,
    ldpc_rate: float,
) -> tuple[int, dict]:
    """Max source payload bits guaranteed to fit channel uses after LDPC+mod padding."""
    mod_bits = int(round(math.log2(mod_order)))
    n = ldpc_codeword_length
    k = _ldpc_k_from_n_rate(n=n, rate=ldpc_rate)
    max_blocks = (channel_uses * mod_bits) // n
    max_source_bits = max_blocks * k
    details = {"k": k, "n": n, "mod_bits": mod_bits, "max_blocks": int(max_blocks)}
    return int(max_source_bits), details


def _tensor_to_u8(image01: tf.Tensor) -> tf.Tensor:
    return tf.image.convert_image_dtype(tf.clip_by_value(image01, 0.0, 1.0), tf.uint8)


def _psnr(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    return float(tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0)).numpy())


def _mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> float:
    return float(tf.reduce_mean(tf.abs(y_true - y_pred)).numpy())


def _bytes_to_bits(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    padded = bits
    rem = bits.size % 8
    if rem != 0:
        padded = np.concatenate([bits, np.zeros(8 - rem, dtype=np.uint8)])
    return np.packbits(padded, bitorder="big").tobytes()


def _modulate(bits: np.ndarray, mod_order: int) -> np.ndarray:
    if mod_order == 2:
        # BPSK: 0 -> +1, 1 -> -1
        return (1.0 - 2.0 * bits.astype(np.float64)).astype(np.complex128)
    if mod_order == 4:
        # QPSK Gray: 00,01,11,10
        rem = bits.size % 2
        if rem != 0:
            bits = np.concatenate([bits, np.zeros(1, dtype=np.uint8)])
        pairs = bits.reshape(-1, 2)
        i = 1.0 - 2.0 * pairs[:, 0].astype(np.float64)
        q = 1.0 - 2.0 * pairs[:, 1].astype(np.float64)
        return (i + 1j * q) / math.sqrt(2.0)
    raise ValueError("real-ldpc mode supports mod_order 2 (BPSK) or 4 (QPSK).")


def _demodulate_llr(symbols_rx: np.ndarray, mod_order: int, noise_var: float) -> np.ndarray:
    if mod_order == 2:
        return (2.0 * np.real(symbols_rx) / max(noise_var, 1e-12)).astype(np.float64)
    if mod_order == 4:
        scale = 2.0 / max(noise_var, 1e-12)
        llr_i = scale * np.real(symbols_rx) * math.sqrt(2.0)
        llr_q = scale * np.imag(symbols_rx) * math.sqrt(2.0)
        return np.stack([llr_i, llr_q], axis=1).reshape(-1).astype(np.float64)
    raise ValueError("real-ldpc mode supports mod_order 2 (BPSK) or 4 (QPSK).")


def _awgn(symbols: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Complex AWGN channel with unit average symbol energy."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_linear
    sigma = math.sqrt(noise_var / 2.0)
    noise = rng.normal(0.0, sigma, size=symbols.shape) + 1j * rng.normal(0.0, sigma, size=symbols.shape)
    return symbols + noise, noise_var


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
    best_fit = None
    for quality in range(100, 0, -1):
        encoded = tf.io.encode_jpeg(image_u8, quality=quality).numpy()
        bits = len(encoded) * 8
        if bits <= target_bits:
            best_fit = (encoded, bits, quality)
            break
    if best_fit is not None:
        return best_fit
    encoded = tf.io.encode_jpeg(image_u8, quality=1).numpy()
    return encoded, len(encoded) * 8, 1


def _compress_bpg_to_budget_bits(
    image_u8: tf.Tensor,
    target_bits: int,
    bpgenc_path: str,
    q_min: int,
    q_max: int,
) -> tuple[bytes, int, int]:
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


def compress_image_to_payload(
    image01: tf.Tensor,
    codec: str,
    target_bits: int,
    bpg_q_min: int,
    bpg_q_max: int,
) -> tuple[bytes, dict]:
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
        codec_used = "bpg"
    else:
        payload, source_bits, quality = _compress_jpeg_to_budget_bits(image_u8, target_bits=target_bits)
        codec_used = "jpeg"

    return payload, {"codec_used": codec_used, "source_bits": int(source_bits), "quality": int(quality)}


def decode_payload_to_image(payload: bytes, codec_used: str) -> tf.Tensor:
    if codec_used == "bpg":
        bpgdec = shutil.which("bpgdec")
        if not bpgdec:
            raise RuntimeError("Requested BPG decode but bpgdec not found on PATH.")
        return _decode_bpg_bytes(payload, bpgdec_path=bpgdec)
    if codec_used == "jpeg":
        return _decode_jpeg_bytes(payload)
    raise ValueError(f"Unsupported codec_used '{codec_used}'.")


def _compress_and_reconstruct(
    image01: tf.Tensor,
    codec: str,
    target_bits: int,
    bpg_q_min: int,
    bpg_q_max: int,
) -> tuple[tf.Tensor, dict]:
    payload, info = compress_image_to_payload(
        image01=image01,
        codec=codec,
        target_bits=target_bits,
        bpg_q_min=bpg_q_min,
        bpg_q_max=bpg_q_max,
    )
    recon = decode_payload_to_image(payload, info["codec_used"])
    return recon, info


def simulate_real_ldpc_link(
    payload: bytes,
    channel_uses: int,
    ldpc_rate: float,
    mod_order: int,
    snr_db: float,
    codeword_length: int,
    row_weight: int,
    bp_iters: int,
    seed: int,
) -> tuple[bool, bytes, dict]:
    bits = _bytes_to_bits(payload)
    code = build_systematic_ldpc(n=codeword_length, rate=ldpc_rate, row_weight=row_weight, seed=seed)
    k, n = code.k, code.n
    mod_bits = int(round(math.log2(mod_order)))

    src_pad = (-bits.size) % k
    if src_pad:
        bits_padded = np.concatenate([bits, np.zeros(src_pad, dtype=np.uint8)])
    else:
        bits_padded = bits
    u_blocks = bits_padded.reshape(-1, k)

    c_blocks = encode_blocks(u_blocks, code)  # (B,n)
    c_bits = c_blocks.reshape(-1)

    mod_pad = (-c_bits.size) % mod_bits
    if mod_pad:
        tx_bits = np.concatenate([c_bits, np.zeros(mod_pad, dtype=np.uint8)])
    else:
        tx_bits = c_bits

    tx_symbols = _modulate(tx_bits, mod_order=mod_order)
    if tx_symbols.size > channel_uses:
        return False, b"", {
            "reason": "channel_use_budget_exceeded",
            "coded_bits": int(c_bits.size),
            "symbols_used": int(tx_symbols.size),
            "block_success_rate": 0.0,
        }

    rng = np.random.default_rng(seed)
    rx_symbols, noise_var = _awgn(tx_symbols, snr_db=snr_db, rng=rng)
    llr = _demodulate_llr(rx_symbols, mod_order=mod_order, noise_var=noise_var)
    llr = llr[: c_bits.size]
    llr_blocks = llr.reshape(-1, n)
    u_hat, ok_mask = decode_blocks_min_sum(llr_blocks, code, max_iter=bp_iters)

    recovered_bits = u_hat.reshape(-1)[: bits.size]
    recovered_payload = _bits_to_bytes(recovered_bits)
    bit_errors = int(np.sum(bits ^ recovered_bits))
    all_ok = bool(np.all(ok_mask) and bit_errors == 0)
    return all_ok, recovered_payload, {
        "reason": "decoded" if all_ok else "decode_failed_or_bit_errors",
        "coded_bits": int(c_bits.size),
        "symbols_used": int(tx_symbols.size),
        "block_success_rate": float(np.mean(ok_mask)) if ok_mask.size else 0.0,
        "bit_errors": bit_errors,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traditional baseline: BPG/JPEG + link model")
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
    p.add_argument("--link-model", choices=["ideal", "real-ldpc"], default="ideal")
    p.add_argument("--ldpc-backend", choices=["custom", "sionna"], default="custom")
    p.add_argument("--ldpc-codeword-length", type=int, default=512)
    p.add_argument("--ldpc-row-weight", type=int, default=3)
    p.add_argument("--ldpc-iters", type=int, default=30)
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
    target_bpp = effective_source_bits_budget / image_pixels
    target_compression_ratio = raw_bits_per_image / max(1, effective_source_bits_budget)
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
        payload, info = compress_image_to_payload(
            image01=image,
            codec=args.codec,
            target_bits=effective_source_bits_budget,
            bpg_q_min=args.bpg_q_min,
            bpg_q_max=args.bpg_q_max,
        )
        used_codec_counts[info["codec_used"]] += 1

        link_info = {}
        if args.link_model == "real-ldpc" and info["source_bits"] > effective_source_bits_budget:
            success = False
            link_info = {
                "reason": "compression_budget_unreachable",
                "source_bits": info["source_bits"],
                "effective_source_bits_budget": effective_source_bits_budget,
            }
            coded_rate_bpcu = (info["source_bits"] / args.ldpc_rate) / args.channel_uses
            recon = tf.zeros_like(image)
        elif args.link_model == "ideal":
            coded_rate_bpcu = (info["source_bits"] / args.ldpc_rate) / args.channel_uses
            success = coded_rate_bpcu <= capacity_bpcu
            if success:
                recon = decode_payload_to_image(payload, info["codec_used"])
            else:
                recon = tf.zeros_like(image)
        else:
            if args.ldpc_backend == "sionna":
                success, rx_payload, link_info = simulate_real_ldpc_link_sionna(
                    payload=payload,
                    channel_uses=args.channel_uses,
                    ldpc_rate=args.ldpc_rate,
                    mod_order=args.mod_order,
                    snr_db=args.snr_db,
                    codeword_length=args.ldpc_codeword_length,
                    bp_iters=args.ldpc_iters,
                    seed=args.split_seed + i,
                )
            else:
                success, rx_payload, link_info = simulate_real_ldpc_link(
                    payload=payload,
                    channel_uses=args.channel_uses,
                    ldpc_rate=args.ldpc_rate,
                    mod_order=args.mod_order,
                    snr_db=args.snr_db,
                    codeword_length=args.ldpc_codeword_length,
                    row_weight=args.ldpc_row_weight,
                    bp_iters=args.ldpc_iters,
                    seed=args.split_seed + i,
                )
            coded_rate_bpcu = link_info.get("coded_bits", 0) / max(1, args.channel_uses)
            if success:
                try:
                    recon = decode_payload_to_image(rx_payload, info["codec_used"])
                except Exception:
                    success = False
                    link_info["reason"] = "decode_payload_failed"
                    recon = tf.zeros_like(image)
            else:
                recon = tf.zeros_like(image)

        if not success:
            outages += 1

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
                "link_info": link_info,
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
        "effective_source_bits_budget": effective_source_bits_budget,
        "real_ldpc_fit_source_bits_max": ldpc_fit_bits,
        "real_ldpc_fit_details": ldpc_fit_details,
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
        "link_model": args.link_model,
        "ldpc_backend": args.ldpc_backend,
        "ldpc_impl": {
            "codeword_length": args.ldpc_codeword_length,
            "row_weight": args.ldpc_row_weight,
            "bp_iters": args.ldpc_iters,
        },
        "output_dir": str(output_dir.resolve()),
        "assumptions": {
            "ideal": "Success when coded spectral efficiency <= estimated channel capacity.",
            "real_ldpc": "Custom sparse systematic LDPC with min-sum decoding; currently AWGN only.",
            "real_ldpc_sionna": "Sionna 5G LDPC backend, when --ldpc-backend sionna is selected.",
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
