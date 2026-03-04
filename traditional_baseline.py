#!/usr/bin/env python3
"""Placeholder CLI for future BPG+LDPC baseline integration."""

from __future__ import annotations

import argparse


def main():
    p = argparse.ArgumentParser(description="Future BPG+LDPC baseline runner")
    p.add_argument("--channel-type", choices=["none", "awgn", "rayleigh", "rician"], default="awgn")
    p.add_argument("--snr-db", type=float, default=10.0)
    p.add_argument("--channel-uses", type=int, default=256)
    p.add_argument("--bpg-quality", type=int, default=30)
    p.add_argument("--ldpc-rate", type=float, default=0.5)
    args = p.parse_args()

    raise NotImplementedError(
        "BPG+LDPC baseline is planned but not implemented yet. "
        f"Requested: channel={args.channel_type}, snr_db={args.snr_db}, "
        f"channel_uses={args.channel_uses}, bpg_quality={args.bpg_quality}, ldpc_rate={args.ldpc_rate}."
    )


if __name__ == "__main__":
    main()
