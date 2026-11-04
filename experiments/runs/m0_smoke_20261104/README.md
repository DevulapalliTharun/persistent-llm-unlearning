# M0 interactive smoke test — 2026-11-04

## Status

Passed interactively through VS Code connected to a Kaggle T4 x2 Jupyter runtime. This is verified M0 smoke-test evidence, not a training result. A committed Kaggle notebook rerun is still required before closing the M0 stage gate.

## Command source

- Runner: `src/m0_smoke_test.py`
- Notebook: `experiments/notebooks/00_m0_smoke_test.ipynb`
- Git commit executed: `d2e9937d1222c0651ba0ecbf8a42096a9f059ae9`
- Base model: `Qwen/Qwen3-0.6B`
- Resolved model revision: `c1899de289a04d12100db370d81485cdf75e47ca`

## Evidence

- CUDA available on two Tesla T4 GPUs.
- Tokenizer chat template was present and non-thinking mode was applied.
- Model loaded in BF16 and produced a non-empty, instruction-consistent response.
- Exact structured output is stored in `result.json`.
- Verified direct dependencies are pinned in `requirements/m0-smoke.txt`.

The original remote `environment.freeze.txt` was generated in `/kaggle/working/m0-smoke/` but has not yet been copied into this Git record. The key runtime and direct dependency versions reported by the run are preserved in `result.json` and `requirements/m0-smoke.txt`.
