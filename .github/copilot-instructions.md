# Copilot instructions for DeepJSCC for EuroSAT RGB

Be concise. Focus on making edits that align with existing project structure and CLI behavior.

What this repo is: short summary
- Purpose: TensorFlow DeepJSCC autoencoder and matching traditional baseline for 64x64 EuroSAT RGB tiles.
- Core pieces: model (in [deepjscc/model.py](deepjscc/model.py)), dataset & sampling ([deepjscc/data.py](deepjscc/data.py)), channel sims ([deepjscc/channels.py](deepjscc/channels.py)), LDPC utilities ([deepjscc/ldpc_codec.py](deepjscc/ldpc_codec.py)) and Sionna glue ([deepjscc/sionna_link.py](deepjscc/sionna_link.py)).

Big-picture architecture notes
- One-process CLI tools (top-level scripts): [run_deepjscc.py](run_deepjscc.py) (train/eval/sample), [traditional_baseline.py](traditional_baseline.py) (digital baseline), and [compare_pipelines.py](compare_pipelines.py) (side-by-side comparisons).
- DeepJSCC is a Keras subclassed model (`DeepJSCC`) assembled in `deepjscc/model.py`. We build the model, call it once with a zero tensor to initialize, then load weights.
- Channel simulation is done in `deepjscc/channels.py`. Supported types: `none`, `awgn`, `rayleigh`, `rician`. Rayleigh/Rician implementations perform an instantaneous per-symbol equalization (division by channel `h`) in the simulator — tests and metrics expect this behaviour.
- Traditional baseline: compress with BPG (fallback to JPEG), then map to LDPC+modulation. Two LDPC backends: in-repo custom (`deepjscc/ldpc_codec.py`) and Sionna (`deepjscc/sionna_link.py`). `real-ldpc` link-model works for AWGN and mod-order 2 or 4 only.

Developer workflows (concrete commands)
- Install: `pip install -r requirements.txt`. For Sionna or GPU add respective req files as in README.
- Train example (from README):
  `python run_deepjscc.py train --image-size 64 --batch-size 64 --epochs 50 --model-variant base --channel-type awgn --snr-db 10 --channel-uses 256 --latent-channels 128 --output-dir artifacts/deepjscc_awgn_snr10 --require-gpu --mixed-precision`
- Evaluate/sample examples: use `python run_deepjscc.py evaluate ...` and `python run_deepjscc.py sample --weights <path>`; `sample` writes `originals/`, `reconstructions/`, `comparisons/`, and `manifest.json`.
- Traditional baseline and comparison: see README examples for `python traditional_baseline.py ...` and `python compare_pipelines.py ...`.

Project-specific conventions & patterns
- CLI-first: prefer adding flags to existing top-level parsers rather than new ad-hoc scripts. Subcommands (`train`, `evaluate`, `sample`) follow a shared `add_shared` pattern in `run_deepjscc.py`.
- Model initialization: always call the model once with a zero batch to build shapes before loading weights (see `build_model()` usage).
- Channel equalization is performed inside the simulator; when adding new channel types, follow `_to_complex`/`_to_ri` helpers and preserve dtype casting.
- LDPC flows: the repo computes an `effective_source_bits_budget` to avoid images that cannot be compressed to the budget — keep this logic when modifying compression or link code.
- BPG binaries: code checks `shutil.which('bpgenc')` / `bpgdec` and falls back to JPEG automatically; keep this graceful fallback.

Integration points & external deps
- TensorFlow 2.x: model, training, and image I/O. Mixed-precision via Keras policy.
- Optional: Sionna (5G) for LDPC backend. If `--ldpc-backend sionna` is requested, scripts raise a helpful error if Sionna is missing.
- Optional external binaries: `bpgenc`/`bpgdec` if `--codec bpg` is used.

Files to inspect when changing behavior
- `run_deepjscc.py` — CLI entry + model lifecycle
- `deepjscc/model.py` — model variants and `MODEL_VARIANTS` keys
- `deepjscc/data.py` — dataset building, TFDS/local fallback behavior
- `deepjscc/channels.py` — channel sims and normalization
- `traditional_baseline.py` & `deepjscc/ldpc_codec.py` — compression, LDPC encode/decode and real-link simulation
- `deepjscc/sionna_link.py` — Sionna integration for LDPC

When uncertain, ask the user for:
- which CLI flags to expose or default changes to make
- whether to prefer Sionna or the custom LDPC for tests/workflows

If you update examples or defaults, also update `README.md` accordingly.

Please review — tell me if you'd like more detail on any area (model internals, LDPC, or data paths).
