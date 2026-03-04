# DeepJSCC for EuroSAT RGB (TensorFlow)

This repository provides a command-line DeepJSCC baseline to reconstruct **64x64 RGB** Earth Observation image tiles from `eurosat/rgb` under configurable wireless channels.

## Features

- TensorFlow DeepJSCC autoencoder for 64x64 RGB tiles.
- Differentiable channel simulation options:
  - `none`
  - `awgn`
  - `rayleigh` (with perfect equalization in the simulator)
  - `rician` (with configurable K-factor and perfect equalization)
- CLI-configurable key parameters, including:
  - `--channel-type`
  - `--snr-db`
  - `--channel-uses`
  - `--latent-channels`
  - train/val/test TFDS splits

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python run_deepjscc.py train \
  --image-size 64 \
  --batch-size 64 \
  --epochs 50 \
  --channel-type awgn \
  --snr-db 10 \
  --channel-uses 256 \
  --latent-channels 128 \
  --output-dir artifacts/deepjscc_awgn_snr10
```

## Evaluate

```bash
python run_deepjscc.py evaluate \
  --image-size 64 \
  --batch-size 64 \
  --channel-type rayleigh \
  --snr-db 5 \
  --weights artifacts/deepjscc_awgn_snr10/best.weights.h5
```

## Notes on future BPG+LDPC comparison

A full BPG+LDPC pipeline is not implemented yet, but the current CLI and channel configuration were designed so a traditional baseline can later be evaluated under the same:

- channel type,
- SNR,
- and channel-use budget.

A starter placeholder command is included in `traditional_baseline.py`.
