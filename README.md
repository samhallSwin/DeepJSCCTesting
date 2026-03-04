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
- Random image sampling utility to save original/reconstruction JPEGs for visual comparison.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GPU setup (optional)

If you want GPU training, install GPU dependencies in addition to the base requirements:

```bash
pip install -r requirements.txt -r requirements-gpu.txt
```

Then verify TensorFlow sees your GPU:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

The CLI also supports:

- `--require-gpu`: fail fast if TensorFlow does not detect a GPU
- `--mixed-precision`: enable `mixed_float16` for faster training on modern GPUs

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
  --output-dir artifacts/deepjscc_awgn_snr10 \
  --require-gpu \
  --mixed-precision
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
  --output-dir artifacts/deepjscc_awgn_snr10 \
  --require-gpu \
  --mixed-precision
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

## Save random visual comparisons

You can sample random images from the full dataset (local EuroSAT folder if available, otherwise TFDS), run them through a trained model, and save JPEG outputs for side-by-side inspection.

```bash
python run_deepjscc.py sample \
  --image-size 64 \
  --channel-type rayleigh \
  --snr-db 5 \
  --weights artifacts/deepjscc_awgn_snr10/best.weights.h5 \
  --num-images 8 \
  --seed 42 \
  --output-dir artifacts/deepjscc_samples
```

Key options:

- `--num-images` number of random samples to save (default `8`)
- `--seed` random seed for reproducible sampling
- `--output-dir` output folder (default `artifacts/deepjscc_samples`)

Output structure:

- `originals/` original input tiles as JPEG
- `reconstructions/` model outputs as JPEG
- `comparisons/` side-by-side `original | reconstruction` JPEG pairs
- `manifest.json` run metadata

## Notes on future BPG+LDPC comparison

A full BPG+LDPC pipeline is not implemented yet, but the current CLI and channel configuration were designed so a traditional baseline can later be evaluated under the same:

- channel type,
- SNR,
- and channel-use budget.

A starter placeholder command is included in `traditional_baseline.py`.
