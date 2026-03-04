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


## Prefer local EuroSAT_RGB dataset (recommended)

If you already have EuroSAT RGB extracted as class folders of JPG files, the loader now checks these paths first (in order):

- `../datasets/EuroSAT_RGB`
- `../../datasets/EuroSAT_RGB`
- `datasets/EuroSAT_RGB`

If any exists, it is used instead of TFDS download. You can also set an explicit path with `--local-eurosat-dir`.

Example:

```bash
python run_deepjscc.py train \
  --image-size 64 \
  --batch-size 64 \
  --epochs 2 \
  --channel-type awgn \
  --snr-db 10 \
  --channel-uses 256 \
  --latent-channels 128 \
  --local-eurosat-dir ../datasets/EuroSAT_RGB \
  --output-dir artifacts/deepjscc_awgn_snr10
```

For local-folder mode, splits are controlled by:

- `--local-train-fraction` (default `0.8`)
- `--local-val-fraction` (default `0.1`)
- test fraction is the remainder
- `--split-seed` controls deterministic split shuffling

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
