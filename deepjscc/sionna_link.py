"""Optional Sionna-based LDPC link simulation helpers."""

from __future__ import annotations

import math

import numpy as np


def _bytes_to_bits(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big")


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    rem = bits.size % 8
    if rem != 0:
        bits = np.concatenate([bits, np.zeros(8 - rem, dtype=np.uint8)])
    return np.packbits(bits, bitorder="big").tobytes()


def _k_from_n_rate(n: int, rate: float) -> int:
    k = int(round(n * rate))
    return max(1, min(k, n - 1))


def simulate_real_ldpc_link_sionna(
    payload: bytes,
    channel_uses: int,
    ldpc_rate: float,
    mod_order: int,
    snr_db: float,
    codeword_length: int,
    bp_iters: int,
    seed: int,
) -> tuple[bool, bytes, dict]:
    """
    Simulate a digital link with Sionna (5G LDPC + QAM mapper/demapper + AWGN).
    Currently AWGN only, and practical for mod_order in {2, 4}.
    """
    try:
        import tensorflow as tf
        import sionna as sn
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "Sionna backend requested but import failed. Install Sionna and compatible TensorFlow."
        ) from exc

    if mod_order not in (2, 4):
        raise ValueError("Sionna backend currently supports mod_order 2 or 4.")

    bits = _bytes_to_bits(payload)
    n = int(codeword_length)
    k = _k_from_n_rate(n=n, rate=ldpc_rate)
    m = int(round(math.log2(mod_order)))

    # Pad info bits to full LDPC blocks.
    src_pad = (-bits.size) % k
    if src_pad:
        bits_padded = np.concatenate([bits, np.zeros(src_pad, dtype=np.uint8)])
    else:
        bits_padded = bits
    u = bits_padded.reshape(-1, k).astype(np.float32)

    tf.random.set_seed(seed)
    encoder = sn.phy.fec.ldpc.LDPC5GEncoder(k=k, n=n, num_bits_per_symbol=m)
    decoder = sn.phy.fec.ldpc.LDPC5GDecoder(encoder=encoder, num_iter=bp_iters, return_infobits=True)
    mapper = sn.phy.mapping.Mapper("qam", num_bits_per_symbol=m)
    demapper = sn.phy.mapping.Demapper("app", "qam", num_bits_per_symbol=m)
    awgn = sn.phy.channel.AWGN()

    c = encoder(tf.convert_to_tensor(u, dtype=tf.float32)).numpy().astype(np.uint8).reshape(-1)
    mod_pad = (-c.size) % m
    if mod_pad:
        c_tx = np.concatenate([c, np.zeros(mod_pad, dtype=np.uint8)])
    else:
        c_tx = c

    symbols_used = c_tx.size // m
    if symbols_used > channel_uses:
        return False, b"", {
            "reason": "channel_use_budget_exceeded",
            "coded_bits": int(c.size),
            "symbols_used": int(symbols_used),
            "block_success_rate": 0.0,
            "backend": "sionna",
        }

    # Map and send through AWGN with unit-power constellation assumption.
    c_tf = tf.convert_to_tensor(c_tx.reshape(1, -1), dtype=tf.float32)
    x = mapper(c_tf)
    snr_linear = 10.0 ** (snr_db / 10.0)
    no = tf.constant(1.0 / snr_linear, dtype=tf.float32)
    y = awgn(x, no)
    llr = demapper(y, no).numpy().reshape(-1)
    llr = llr[: c.size].reshape(-1, n)

    u_hat = decoder(tf.convert_to_tensor(llr, dtype=tf.float32)).numpy()
    u_hat = (u_hat > 0.5).astype(np.uint8).reshape(-1)
    bits_hat = u_hat[: bits.size]
    payload_hat = _bits_to_bytes(bits_hat)

    bit_errors = int(np.sum(bits ^ bits_hat))
    success = bit_errors == 0
    return success, payload_hat, {
        "reason": "decoded" if success else "decode_failed_or_bit_errors",
        "coded_bits": int(c.size),
        "symbols_used": int(symbols_used),
        "block_success_rate": float(1.0 if success else 0.0),
        "bit_errors": bit_errors,
        "backend": "sionna",
        "k": int(k),
        "n": int(n),
    }
