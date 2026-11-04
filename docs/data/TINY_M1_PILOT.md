# Tiny M1 learning diagnostic

**Status:** Implemented; CPU data preparation verified. PEFT import, CUDA training, learning and adapter persistence are not yet tested.

This is the first training experiment. It checks whether the candidate model, tokenizer, answer masking, LoRA updates and artifact workflow can learn a small set of fictional facts. Its output is a diagnostic adapter, not the canonical M1 used for the dissertation comparisons.

## Scope and bounded dataset review

Use the first four F units and first eight R units in sorted record-ID order from the unchanged seed-42 candidate. Selection does not depend on model performance. The fixed IDs and candidate checksum are in `configs/training/m1_tiny.json`.

| Partition | IDs |
|---|---|
| F | `ku_000001`, `ku_000008`, `ku_000009`, `ku_000010` |
| R | `ku_000002` through `ku_000007`, plus `ku_000011`, `ku_000012` |

The 12 canonical author/book/prize/year records were manually inspected, along with the existing task/template definitions. The selected mappings are coherent and have unique named entities/paths under the candidate's checked invariants. Existing full-candidate generation checks, independent byte reproduction, near-duplicate/token-volume audits and verified M0 baselines are preserved. Limited external phrase searches for this selection were inconclusive because results included unrelated keyword matches; they do not certify that every surface is absent from public sources.

The selected subset is admitted for this limited infrastructure/learning diagnostic. This is not full-dataset acceptance or a claim of exhaustive real-world name clearance. The full candidate stays under `data/candidates/`; broader manual review and final release decisions remain pending. No facts or final templates are regenerated based on M0 scores.

There are 84 training rows and 84 development rows (seven task families per unit). Only `m1_train` and `m1_dev` supply model examples. N_rel, N_reason and final-test examples are excluded. The development prompts use held-out wording over learned units, not unseen units. Because composite task answers are also present in training, passing this diagnostic establishes access under new wording, not independently demonstrated compositional reasoning.

## Training representation

Use the same `answer-only-v2` system instruction and chat template as the verified baseline. Render assistant targets as `Answer: <canonical answer>`.

`src/training/data.py` verifies that the full training token sequence begins with the exact inference prompt tokens. Prompt tokens receive label `-100`; answer tokens and the first assistant EOS are supervised. Crop only trailing post-EOS template tokens, and refuse sequence-length truncation. Right padding receives zero attention and ignored labels, even if the pad ID equals the EOS ID.

CPU preparation with the pinned Qwen tokenizer passed for all 84 training rows: maximum length 101 tokens, 7,153 prompt tokens, 601 supervised answer/EOS tokens, zero truncated examples. These are preparation measurements, not training results.

## Fixed pilot configuration

- Pinned Qwen3-0.6B and tokenizer revision, seed 42; CUDA FP16 base weights on one GPU.
- LoRA rank 8, alpha 16, dropout 0, all linear layers, no bias training.
- Only FP32 LoRA parameters are allowed in the optimizer. Base parameters remain frozen.
- AdamW learning rate 0.0003, no weight decay, gradient clipping 1.0.
- Batch size 4, gradient accumulation 4, 120 optimizer attempts.
- Answer-token CE averaged per example before averaging the batch.
- AMP scaling is logged; skipped updates are counted and prevent claiming the full optimizer budget was applied.
- Same v2 greedy/non-thinking 128-token generation for base, trained and reloaded evaluations.

PEFT 0.21.2 is pinned in `requirements/m1-tiny.txt`; its package metadata allows the existing pinned dependencies. Actual runtime import/training compatibility still requires the Kaggle run. Adapter creation and restoration follow the [official PEFT guide](https://huggingface.co/docs/peft/quicktour).

## Gates defined before the run

The pilot passes only if:

1. In each of F and R, direct author↔book development accuracy is at least 90%, and composite two-/three-edge task accuracy is at least 75%.
2. The held-out ten-question arithmetic probe has base accuracy at least 80%; post-training accuracy drops by at most 10 percentage points. This is a small smoke check, not a general reasoning benchmark.
3. Development and utility outputs have zero label-format violations and zero token-limit hits.
4. LoRA parameters changed, and all 120 optimizer updates were applied.
5. Loading the saved adapter onto a newly instantiated pinned base reproduces all trained-model development and utility strings exactly.

These are diagnostic go/no-go thresholds, not final M1/M2 research thresholds. A failed pilot is a useful result; preserve it and diagnose with development data. Do not tune against final-test outputs. Full M1 and M2 remain future stages.

## Files and persistence

Use `experiments/notebooks/03_m1_tiny_learning.ipynb` from VS Code with the Kaggle kernel. It requires the existing `HF_TOKEN` attached through Kaggle Secrets because this experiment creates adapter files that must be preserved privately.

The notebook checks private repository access before training. It uploads the diagnostic run under:

```text
tharundevulapalli/persistent-llm-unlearning-checkpoints
  diagnostics/m1-tiny/<unique-run-name>/
```

It then downloads the adapter files at the upload's immutable revision, compares hashes, and saves/uploads a persistence receipt. No complete base-model checkpoint is saved or uploaded.

The run contains an adapter, tokenizer settings, final optimizer/scaler snapshot, selected input rows, per-attempt CE logs, before/after/reloaded predictions, gate results, package versions and provenance. The optimizer snapshot is for inspection/recovery work; this runner does not implement exact mid-run resume or claim to save every RNG/data-loader state.

The final notebook cell exports a ZIP containing only small JSON/JSONL/text evidence for Git. Model/optimizer binaries remain in private Hugging Face storage. Keep failed-run evidence too. A completed script is distinct from passing learning gates and from verified persistent storage.

## Local commands

Plan without model/tokenizer downloads:

```bash
python3 -m src.training.tiny_m1 --config configs/training/m1_tiny.json \
  --output-dir /tmp/m1-tiny-plan --mode plan
```

Tokenizer-only preparation (requires `requirements/data-audit.txt`):

```bash
python -m src.training.tiny_m1 --config configs/training/m1_tiny.json \
  --output-dir /tmp/m1-tiny-prepare --mode prepare
```

The notebook's explicitly labelled training cell uses `--mode train` on Kaggle. Choose a new empty output directory for each invocation; existing results are not overwritten.
