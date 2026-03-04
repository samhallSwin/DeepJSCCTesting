"""Simple sparse LDPC codec with min-sum decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimpleLDPC:
    n: int
    k: int
    p: np.ndarray  # shape: (k, m)
    h: np.ndarray  # shape: (m, n)
    check_neighbors: list[np.ndarray]
    var_neighbors: list[np.ndarray]


def build_systematic_ldpc(
    n: int,
    rate: float,
    row_weight: int = 3,
    seed: int = 42,
) -> SimpleLDPC:
    """Build a sparse systematic LDPC code with H = [P^T | I]."""
    if n <= 2:
        raise ValueError("n must be > 2.")
    if not (0.0 < rate < 1.0):
        raise ValueError("rate must be in (0, 1).")
    if row_weight <= 0:
        raise ValueError("row_weight must be > 0.")

    k = int(round(n * rate))
    k = max(1, min(k, n - 1))
    m = n - k

    rng = np.random.default_rng(seed)
    p = np.zeros((k, m), dtype=np.uint8)
    w = min(row_weight, m)
    for i in range(k):
        cols = rng.choice(m, size=w, replace=False)
        p[i, cols] = 1

    # H shape: (m, n)
    h = np.concatenate([p.T, np.eye(m, dtype=np.uint8)], axis=1)

    check_neighbors = [np.where(h[j] == 1)[0] for j in range(m)]
    var_neighbors = [np.where(h[:, i] == 1)[0] for i in range(n)]

    return SimpleLDPC(
        n=n,
        k=k,
        p=p,
        h=h,
        check_neighbors=check_neighbors,
        var_neighbors=var_neighbors,
    )


def encode_blocks(info_bits: np.ndarray, code: SimpleLDPC) -> np.ndarray:
    """Encode info bits in blocks. info_bits shape: (B, k), returns (B, n)."""
    if info_bits.ndim != 2 or info_bits.shape[1] != code.k:
        raise ValueError(f"info_bits must have shape (B, {code.k}).")
    parity = (info_bits @ code.p) % 2
    return np.concatenate([info_bits, parity.astype(np.uint8)], axis=1).astype(np.uint8)


def syndrome(bits: np.ndarray, code: SimpleLDPC) -> np.ndarray:
    """Return syndrome for codeword bits shape (..., n)."""
    return (bits @ code.h.T) % 2


def decode_block_min_sum(llr: np.ndarray, code: SimpleLDPC, max_iter: int = 30) -> tuple[np.ndarray, bool]:
    """Decode one block via min-sum. llr shape: (n,)."""
    n = code.n
    m = n - code.k
    if llr.shape != (n,):
        raise ValueError(f"llr must have shape ({n},).")

    # Messages on H edges; zeros for absent edges.
    q = np.zeros((m, n), dtype=np.float64)  # var->check
    r = np.zeros((m, n), dtype=np.float64)  # check->var
    # Initialize var->check messages with channel LLR on existing Tanner-graph edges.
    q = code.h.astype(np.float64) * llr[np.newaxis, :]

    hard = np.zeros(n, dtype=np.uint8)
    for _ in range(max_iter):
        # Check node update.
        for j, neigh in enumerate(code.check_neighbors):
            vals = q[j, neigh]
            if vals.size == 0:
                continue
            abs_vals = np.abs(vals)
            signs = np.sign(vals)
            signs[signs == 0] = 1.0
            sign_prod = np.prod(signs)
            min1_idx = int(np.argmin(abs_vals))
            min1 = abs_vals[min1_idx]
            if abs_vals.size > 1:
                min2 = np.partition(abs_vals, 1)[1]
            else:
                min2 = min1

            for t, i in enumerate(neigh):
                mag = min2 if t == min1_idx else min1
                s = sign_prod * signs[t]
                r[j, i] = s * mag

        # Variable node update + hard decision.
        posterior = llr.copy()
        for i, checks in enumerate(code.var_neighbors):
            if checks.size == 0:
                continue
            posterior[i] += np.sum(r[checks, i])
            for j in checks:
                q[j, i] = posterior[i] - r[j, i]
        hard = (posterior < 0).astype(np.uint8)

        if np.all(syndrome(hard[np.newaxis, :], code) == 0):
            return hard, True

    return hard, False


def decode_blocks_min_sum(
    llr_blocks: np.ndarray, code: SimpleLDPC, max_iter: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Decode multiple blocks. Returns decoded_bits (B, k), success_mask (B,)."""
    if llr_blocks.ndim != 2 or llr_blocks.shape[1] != code.n:
        raise ValueError(f"llr_blocks must have shape (B, {code.n}).")

    bsz = llr_blocks.shape[0]
    decoded = np.zeros((bsz, code.k), dtype=np.uint8)
    success = np.zeros((bsz,), dtype=bool)
    for b in range(bsz):
        cw, ok = decode_block_min_sum(llr_blocks[b], code, max_iter=max_iter)
        decoded[b] = cw[: code.k]
        success[b] = ok
    return decoded, success
