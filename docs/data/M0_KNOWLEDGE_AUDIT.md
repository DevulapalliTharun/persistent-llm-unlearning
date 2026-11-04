# M0 candidate knowledge audit

For the next comparison baseline, use `experiments/notebooks/02_m0_answer_only_audit.ipynb` with `configs/evaluation/m0_knowledge_audit_v2.json`. Its answer instruction, 128-token cap, parser and stop-reason logging are documented in `docs/data/EVALUATION_PROTOCOL.md`. V2 is locally tested; GPU results are pending. The original run and protocol below remain preserved.

This audit records how the original Qwen3-0.6B checkpoint responds to the candidate facts before any fact-learning training. On 2026-11-05 IST, original GPU outputs were downloaded and independently verified: 1,260 predictions, zero whole-string exact matches and two year-mention flags, including one semantically correct year in prose. A 42-response sample also exposed truncated-looking answers under the 32-token cap. See `experiments/runs/m0_knowledge_audit_20261105/` for the preserved original evidence and remaining evaluation limitations.

## Inputs and fixed protocol

- Configuration: `configs/evaluation/m0_knowledge_audit.json`.
- Candidate: `data/candidates/pilot-v1/seed-42/`, pinned by the SHA-256 digest of its checksum index.
- Model and tokenizer revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Input files: `m1_dev.jsonl` (840 prompts) and `n_rel_train.jsonl` (420 prompts).
- Coverage: all 180 knowledge units, 7 task types, 1,260 predictions.
- Questions are sent without the answer or supporting graph facts. Final-test prompts are not read.
- Seed 42, non-thinking mode, greedy decoding, 32 new tokens maximum, batch size 16.
- One CUDA GPU; T4 uses FP16. The earlier smoke test reported BF16 through its hardware check; this audit explicitly requires native BF16 capability before selecting BF16 and records its actual dtype.

Greedy decoding is chosen for a repeatable initial audit. The smoke test used sampling; do not compare their outputs as if decoding and precision were identical. Final evaluation decoding and acceptance criteria must be defined separately before training.

The batched loader uses left padding and removes prompt tokens from decoded responses, following the [Transformers generation guide](https://huggingface.co/docs/transformers/llm_tutorial).

## Run from VS Code with the Kaggle kernel

1. Commit and push the candidate files, configuration, runner, notebook and context updates to `main`.
2. Start a Kaggle GPU notebook, enable Internet, and copy its VS Code-compatible Jupyter URL.
3. Open `experiments/notebooks/01_m0_knowledge_audit.ipynb` locally in VS Code and select that remote Python kernel.
4. Run its five code cells in order: fresh clone, pinned dependencies, checksum preflight, inference, summary.
5. Preserve the output directory printed by the notebook. It is created under `/kaggle/working/m0-knowledge-audit-*`.

The public model does not require the private checkpoint-store token. Training and adapter uploads are not part of this audit.

## Output files

| File | Meaning |
|---|---|
| `plan.json` | Config, exact candidate/input hashes, Git commit and expected counts |
| `predictions.jsonl` | Every question, expected answer, raw generated response and scoring fields |
| `summary.json` | Execution status, counts by partition/task, runtime/dtype, prediction checksum and pending review state |
| `matches_for_review.json` | Matches/mentions to inspect manually |
| `environment.freeze.txt` | Installed package versions |

The script refuses to overwrite a nonempty output directory. It flushes predictions after each batch; an interrupted run may leave a partial prediction file, but a completed summary is written only after all prompts finish. Start a new output directory for a retry and retain the partial record if diagnosing a failure.

Download these small files into a dated `experiments/runs/` folder after execution. For a clean committed rerun, import this notebook into Kaggle and use Save & Run All; retain its outputs. A local VS Code notebook and the Kaggle browser notebook share compute but are separate notebook files.

## Interpret results before dataset acceptance

`normalized_exact_match` uses case folding, whitespace normalization and removal of surrounding punctuation/quotes. It does not silently accept a longer explanatory sentence. `expected_answer_mentioned` additionally flags the expected answer as a complete surface string within a longer response; a mention is not scored as a correct answer.

Inspect all flags and raw predictions. A year can be guessed by chance, so year matches are counted separately. A coincidentally correct name/title also requires review; one answer does not prove that a model memorized the association. Failed responses do not prove knowledge is absent.

Preserve the candidate and audit outputs. Do not automatically regenerate examples to force the baseline accuracy to zero, change final templates based on the audit, or treat the execution flag as dataset acceptance. Record any genuine collision/exclusion decision with evidence and a new candidate version if needed. Full manual name/title collision review remains a separate gate before the tiny M1 pilot.

## CPU preflight only

This verifies the inputs without downloading model weights:

```bash
python3 -m src.m0_knowledge_audit \
  --config configs/evaluation/m0_knowledge_audit.json \
  --output-dir /tmp/m0-knowledge-plan \
  --plan-only
```

Choose a new empty output directory if that path already contains a previous plan.
