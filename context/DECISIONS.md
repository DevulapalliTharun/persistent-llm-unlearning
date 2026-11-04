# Decision Log

This file records what has actually been agreed versus what remains provisional. Update decisions after experiments, rather than silently changing the plan.

## Confirmed project-level decisions

| ID | Decision | Rationale / consequence |
|---|---|---|
| D-01 | Explore **two candidate methods**, RAFG and RASP, addressing the **same benign-relearning problem**. | Share research setup; compare the two approaches after the baseline is demonstrated. |
| D-02 | Build **one common M0 → M1 → M2 pipeline**. | No duplicated fact-training and initial unlearning work. |
| D-03 | Save one canonical, validated **M2** checkpoint and branch from independent copies. | Fair comparison: identical effective starting weights/adapters. |
| D-04 | Run **ordinary, undefended later fine-tuning** as a recovery pilot before claiming a guard is needed. | Demonstrate the failure being addressed. |
| D-05 | Initial unlearning is **existing baseline methodology**, not the project's new algorithm. | Contributions concern persistence after unlearning. |
| D-06 | Preserve all **seven named metrics** in the evaluation plan. | Balance target forgetting, reasoning usefulness, recovery and cost. |
| D-07 | Prioritize **direct/multi-hop forget accuracy, retain accuracy and reasoning baseline** before/at M2. | Later metrics depend on a functioning M2 and continuation setup. |
| D-08 | Prefer **no paid LLM API tokens** and a portable VS Code/Git workflow. | Ability to continue after switching computers. |
| D-09 | Store proposal PDFs under `docs/proposals/`; use `context/` for the current written project state. | Keep proposals separate from operational status and actual results. |
| D-10 | Use local VS Code clients for development, GitHub for code/context, Kaggle GPU as the primary execution environment, Colab as fallback, and private Hugging Face model repository `tharundevulapalli/persistent-llm-unlearning-checkpoints` for durable adapters/checkpoints. | Keeps the workflow portable across the company Mac and personal Dell while separating reviewed source, ephemeral compute and large persistent artifacts. A committed Kaggle-version upload/download checksum round-trip was verified on 2026-11-04; the same rule applies to every important checkpoint. |

## Provisional technical choices — confirm during smoke tests

| ID | Proposed choice | Validation needed |
|---|---|---|
| P-01 | Open model `Qwen/Qwen3-0.6B`. | Model loading, correct chat template, usable direct/multi-hop capacity and available GPU memory. |
| P-02 | LoRA with rank 8 or 16. | GPU fit and knowledge-learning ability. |
| P-03 | Gradient Difference for M1 → M2: `-alpha*CE(F) + beta*CE(R)`. | Stable forgetting, retain preservation and correct gradient sign/masking. |
| P-04 | Fictional author/book/prize/year data; initial ~120 entities, 30 forget, 90 retain. | Prompt diversity, answer uniqueness, learning feasibility and leakage audit. |
| P-05 | Two later-task conditions: unrelated reasoning and similarly structured **new** relational data. | Whether they yield natural, nontrivial target recovery. |
| P-06 | RAFG: M2-relative CE monitoring + triggered margin-based correction. | Threshold calibration, monitoring cost, false positives and baseline comparisons. |
| P-07 | RASP: forget-vs-retain gradient module ranking + M2-anchored weighted LoRA drift penalty. | Score validity, parameterization, retained learning and baseline comparisons. |
| P-08 | Kaggle remote GPU connected to VS Code as the primary runtime; Colab as fallback. | Runtime availability, permissions, remote-kernel connection, artifact restore, authentication and stable package versions. |

## Explicitly NOT decided yet

- The final model checkpoint/revision, Python/library versions, remote provider and precision/quantization settings.
- Final dataset size, templates, exact entity splits, train/validation/test ratios and seeds.
- The numerical thresholds defining sufficient M1 learning, acceptable M2 forgetting and nontrivial recovery.
- Exact optimizer, learning rates, `alpha`, `beta`, RAFG thresholds/intervals/bursts and RASP `lambda`/module granularity.
- Whether either proposed method will outperform tuned, fair baselines; no gains have been measured.
- Whether both projects will remain separate proposals or one will become the final dissertation focus.

## Change policy

When a choice becomes final, move it to Confirmed with a date and reason. Record superseded decisions instead of erasing history. Add links to experiment configs/results supporting changes.
