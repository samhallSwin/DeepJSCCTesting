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
  - `--model-variant` (`tiny`, `small`, `base`, `large`)
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

Optional Sionna backend for real LDPC:

```bash
pip install -r requirements.txt -r requirements-sionna.txt
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
  --model-variant base \
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
  --model-variant base \
  --channel-type rayleigh \
  --snr-db 5 \
  --weights artifacts/deepjscc_awgn_snr10/best.weights.h5
```

`train` now writes `architecture.txt` into `--output-dir` (alongside `best.weights.h5` and `last.weights.h5`) with:

- selected variant and core hyperparameters
- encoder summary
- decoder summary
- full model summary

When running `evaluate` or `sample`, use the same `--model-variant` used during training so the weight shapes match.

## Save random visual comparisons

You can sample random images from the full dataset (local EuroSAT folder if available, otherwise TFDS), run them through a trained model, and save JPEG outputs for side-by-side inspection.

```bash
python run_deepjscc.py sample \
  --image-size 64 \
  --model-variant base \
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

## Traditional baseline (BPG/JPEG + LDPC-rate-constrained link)

You can run a traditional digital baseline under the same channel setup and channel-use budget as DeepJSCC:

```bash
python traditional_baseline.py \
  --image-size 64 \
  --batch-size 64 \
  --channel-type awgn \
  --snr-db 10 \
  --channel-uses 256 \
  --ldpc-rate 0.5 \
  --mod-order 4 \
  --codec bpg \
  --num-images 512 \
  --output-dir artifacts/traditional_baseline_awgn_snr10
```

Real LDPC mode (custom sparse LDPC + min-sum decoder, AWGN only):

```bash
python traditional_baseline.py \
  --image-size 64 \
  --batch-size 64 \
  --channel-type awgn \
  --snr-db 10 \
  --channel-uses 256 \
  --ldpc-rate 0.5 \
  --mod-order 4 \
  --codec bpg \
  --link-model real-ldpc \
  --ldpc-backend custom \
  --ldpc-codeword-length 512 \
  --ldpc-row-weight 3 \
  --ldpc-iters 30 \
  --num-images 512 \
  --output-dir artifacts/traditional_baseline_real_ldpc
```

Use `--ldpc-backend sionna` to run the same real-LDPC flow with Sionna 5G LDPC blocks (requires Sionna installed).

Key fairness controls:

- `--channel-type`, `--snr-db`, `--rician-k`
- `--channel-uses` (same channel-use budget as DeepJSCC)
- `--ldpc-rate` (source bits = coded bits * rate)
- `--mod-order` (bits per channel use = `log2(mod_order)`)
- `--link-model`:
  - `ideal` (capacity-threshold model)
  - `real-ldpc` (explicit LDPC encode/decode over AWGN; currently `mod-order` 2 or 4)
- `--ldpc-backend` (used when `--link-model real-ldpc`):
  - `custom` (in-repo sparse LDPC + min-sum)
  - `sionna` (Sionna 5G LDPC)

Codec behavior:

- `--codec bpg` uses `bpgenc`/`bpgdec` if found on PATH
- falls back to JPEG automatically if BPG binaries are unavailable

Outputs:

- `summary.json` aggregate metrics and settings
- `per_image.json` per-sample details
- `originals/`, `reconstructions/`, `comparisons/` JPEGs

Modeling note:

- `ideal` link model: decode success if coded spectral efficiency is below estimated channel capacity; otherwise outage.
- `real-ldpc` link model: uses a custom sparse systematic LDPC code with min-sum decoding.
- In `real-ldpc`, the script computes an `effective_source_bits_budget` that accounts for LDPC block sizing and modulation padding, to avoid avoidable `channel_use_budget_exceeded` cases.
- If the codec cannot compress an image down to that effective budget (even at lowest quality), the image is marked failed with reason `compression_budget_unreachable`.

## Compare DeepJSCC vs traditional on same random images

Use `compare_pipelines.py` to sample one random image set, run both pipelines under the same channel conditions, and save side-by-side outputs.

```bash
python compare_pipelines.py \
  --image-size 64 \
  --num-images 8 \
  --seed 42 \
  --channel-type awgn \
  --snr-db 10 \
  --channel-uses 256 \
  --deepjscc-weights artifacts/deepjscc_awgn_snr10/best.weights.h5 \
  --model-variant base \
  --latent-channels 128 \
  --ldpc-rate 0.5 \
  --mod-order 4 \
  --codec bpg \
  --link-model ideal \
  --ldpc-backend custom \
  --output-dir artifacts/pipeline_comparison_awgn_snr10
```

To use explicit LDPC in the comparison script, set:

- `--link-model real-ldpc`
- `--channel-type awgn`
- `--mod-order 2` or `--mod-order 4`
- `--ldpc-backend custom` or `--ldpc-backend sionna`
- optional tuning: `--ldpc-codeword-length`, `--ldpc-row-weight`, `--ldpc-iters`

Output structure:

- `originals/` sampled originals
- `deepjscc/` DeepJSCC reconstructions
- `traditional/` traditional-pipeline reconstructions
- `comparisons/` horizontal triplets: `original | deepjscc | traditional`
- `summary.json` aggregate metrics/settings
- `per_image.json` per-image metrics and traditional link details
