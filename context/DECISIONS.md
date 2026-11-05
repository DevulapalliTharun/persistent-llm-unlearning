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
| D-11 | Represent each synthetic fact as an entity-first author → book → prize → award-year knowledge unit; partition units into `F`, `R`, and disjoint relational `N`, while treating `T` as a held-out prompt view rather than a training entity split. | Preserves unique multi-hop paths, prevents cross-partition entity leakage and keeps final evaluation prompts unavailable to all training, monitoring, scoring and checkpoint-selection steps. Full rules are in `docs/data/DATASET_SPEC.md`. |
| D-12 | Use the versioned `answer-only-v2` wrapper/parser, 128-token cap and per-response EOS/cap logging for the next M0 baseline and subsequent checkpoint comparisons. Preserve v1 evidence and report v2 parsed-answer accuracy separately from legacy whole-response matching and mere mentions. | Defined before M1/final-test evaluation using the original development audit's format/truncation observations. Same system instruction and assistant target format must be used by the future SFT renderer; GPU verification and learning thresholds remain pending. See `docs/data/EVALUATION_PROTOCOL.md`. |

## Provisional technical choices — confirm during smoke tests

| ID | Proposed choice | Validation needed |
|---|---|---|
| P-01 | Open model `Qwen/Qwen3-0.6B`. | Model loading, correct chat template, usable direct/multi-hop capacity and available GPU memory. |
| P-02 | LoRA with rank 8 or 16. | GPU fit and knowledge-learning ability. |
| P-03 | Gradient Difference for M1 → M2: `-alpha*CE(F) + beta*CE(R)`. | Stable forgetting, retain preservation and correct gradient sign/masking. |
| P-04 | Fictional author/book/prize/year data; initial 120 M1 knowledge units, 30 forget and 90 retain; provisional 60 disjoint `N_rel` units. | Generator determinism, prompt diversity, answer uniqueness, learning feasibility, exact `N` budgets and leakage audit. |
| P-05 | Two later-task conditions: unrelated reasoning and similarly structured **new** relational data. | Whether they yield natural, nontrivial target recovery. |
| P-06 | RAFG: M2-relative CE monitoring + triggered margin-based correction. | Threshold calibration, monitoring cost, false positives and baseline comparisons. |
| P-07 | RASP: forget-vs-retain gradient module ranking + M2-anchored weighted LoRA drift penalty. | Score validity, parameterization, retained learning and baseline comparisons. |
| P-08 | Kaggle remote GPU connected to VS Code as the primary runtime; Colab as fallback. | Runtime availability, permissions, remote-kernel connection, artifact restore, authentication and stable package versions. |
| P-09 | Match `N_rel` and `N_reason` with equal rows/update opportunities and no more than 5% total-token difference under the pinned Qwen tokenizer; reject normalized author/book/prize pairs at similarity ≥ 0.92. | Full-pilot tokenizer audit, manual sample review and sensitivity to the chosen thresholds. |
| P-10 | Use one exclusive template per non-final task/partition and two exclusive held-out templates per final-test task. | Tiny review passed; reassess wording breadth after the tiny M1 learning/evaluation pilot. |

## 2026-11-05 — Full candidate evidence

The 30 F / 90 R / 60 N_rel seed-42 candidate is saved in `data/candidates/pilot-v1/seed-42/`, with validation evidence in `experiments/runs/data_full_validation_20261105/`. Independent generations matched byte-for-byte; the full token-volume audit passed at a 2.88% difference. Preserve it as a candidate until full manual collision review and the M0 knowledge audit pass. The candidate path distinguishes generated files from an accepted training release.

The first full generation exposed duplicate arithmetic questions. Repeated questions are now skipped deterministically while keeping the requested number of unique rows. Manifests embed exact configuration and generator-source hashes so the pre-commit correction is traceable alongside the baseline Git commit.

## 2026-11-05 — M0 knowledge-audit protocol

Use the pinned Qwen revision with non-thinking greedy decoding on all 180 candidate units through `m1_dev` and `n_rel_train` prompts (1,260 total). Preserve final-test wording. The audit flags exact matches and answer mentions for manual review, separates year matches, and never automatically edits or accepts the dataset. This avoids filtering the candidate merely to force a zero M0 score. The protocol and fixed config are in `docs/data/M0_KNOWLEDGE_AUDIT.md` and `configs/evaluation/m0_knowledge_audit.json`.

The runner uses native-capability-aware precision selection (FP16 on T4) and records it. Its results must not be interpreted as a direct repeat of the earlier BF16/sampled smoke test. On 2026-11-05 IST the user reported completed inference: 1,260 predictions, zero exact matches and two year mentions in longer responses. Preserve and inspect the raw files before interpreting those flags or accepting the candidate; see `experiments/runs/m0_knowledge_audit_20261105/`.

## 2026-11-05 — Interpretation of year flags

Manual inspection identified an unsupported year mention with incorrect supporting book/prize details and a semantically correct year expressed as a sentence. Preserve the original whole-string exact-match result and report the manual finding separately; zero exact matches is not equivalent to zero correct answers. Do not automatically exclude or regenerate either record. Define consistent answer extraction/formatting rules before M1/M2 comparisons. Evidence: `experiments/runs/m0_knowledge_audit_20261105/match_review.json`.

## 2026-11-05 — Original audit verification and generation limits

All original audit files were imported and independently checked against the candidate and rescored summary. A 42-response sample includes visible mid-sentence endings under the 32-token cap. Preserve this baseline and its original scoring. Before M1/M2 comparisons, define consistent answer extraction, a sufficient generation budget and per-response termination logging; any revised baseline must be recorded separately. Evidence: `experiments/runs/m0_knowledge_audit_20261105/artifact_verification.json`.

## 2026-11-05 — Reported v2 protocol execution

User-reported v2 execution completed all 1,260 prompts with zero parsed-answer matches, format violations and token-limit hits. Record these as observed under the fixed protocol and preserve the baseline; do not interpret them as certified absence of knowledge or automatically accept the dataset. Independent artifact verification and raw-response/abstention sampling remain pending. See `experiments/runs/m0_answer_only_audit_20261105/`.

## 2026-11-05 — Verified v2 baseline interpretation

All original v2 files are now preserved and checked. The model replied `Answer: Unknown.` 1,259 times and supplied one wrong book association; every response logs EOS and the parsed-match/format/cap summary reproduces. Accept this as the baseline behaviour under v2, not evidence that the facts have been erased or are absent internally. Keep the fixed protocol for M1/M2 comparisons and require demonstrated M1 learning before interpreting later suppression. Do not regenerate targets or repeat M0 runs merely to change this baseline. Evidence: `experiments/runs/m0_answer_only_audit_20261105/artifact_verification.json`.

## 2026-11-05 — Bounded tiny M1 diagnostic

Admit only the fixed first four F / first eight R units by sorted record ID for the next learning/infrastructure pilot. Their records and task/template mappings were inspected, automated candidate gates pass, and both M0 baselines are preserved. Limited external phrase searches were inconclusive; do not claim universal name clearance or full-dataset release. Keep the canonical candidate unchanged.

The diagnostic uses 84 train / 84 development rows, rank-8 all-linear LoRA, 120 optimizer attempts, fixed v2 scoring and a fresh-base adapter reload comparison. Before execution, set per-F/R direct author-book accuracy ≥90%, composite task accuracy ≥75%, arithmetic base accuracy ≥80% and post drop ≤10 points, zero format/cap issues, changed LoRA weights, 120 applied updates and exact reload-prediction reproduction as gates. These are pilot thresholds, not final benchmark acceptance thresholds. Details: `docs/data/TINY_M1_PILOT.md`.

## 2026-11-05 — Remove unused optional TorchAO in the Kaggle pilot runtime

Keep PEFT 0.21.2, the pinned M0 stack, model revision, data and training hyperparameters unchanged. The first adapter-construction attempt was blocked by preinstalled TorchAO 0.10.0, which PEFT probes even for unquantized LoRA. Remove that unused package only in the disposable Kaggle runtime and validate the backend in a fresh subprocess. Add an early compatibility check before model loading. No local Mac packages or model/dataset files are removed. Retry in a fresh run directory and retain the failure evidence.

## 2026-11-05 — Preserve failed tiny-learning diagnostic before tuning

The completed retry demonstrates learning and fresh-base reload on the tested subset, but does not pass all predefined gates. Keep `pilot.passed=false`: R composite 16/24 is below 75%, and base utility 0/10 makes the utility baseline unmeasurable. Preserve the adapter/reports as diagnostic evidence, not canonical M1. Inspect raw failed development and utility items before selecting a remedy; do not retrospectively lower thresholds or claim utility preservation from the mechanical nondecrease check. Evidence: `experiments/runs/m1_tiny_attempt_20261105/`.

## 2026-11-05 — Verified tiny M1 diagnosis; retain original scoring

All original tiny M1 small evidence files and the private persistence receipt are preserved. Local rescoring reproduces the failed pilot. Base arithmetic is 10/10 abstentions; post arithmetic includes three correct numbers with units beyond the four strict matches. Three R year outputs also contain correct years with extra text. Keep these as separate manual findings; do not change v2 scores or weaken gates. Genuine relational/arithmetic errors remain. Prompt sensitivity requires a separately versioned, controlled base/adapter diagnostic before further SFT; that comparison is proposed, not executed. No model-size, learning-rate or update-budget change is justified solely by this evidence. See `experiments/runs/m1_tiny_attempt_20261105/manual_review.json`.

## 2026-11-05 — Versioned utility prompt comparison before more training

Implement a separate inference-only diagnostic on the unchanged ten arithmetic questions, pinned Qwen base and privately restored tiny adapter. Compare the exact original-v2 system prompt with a prespecified arithmetic-only integer instruction, holding greedy/non-thinking decoding and precision/settings fixed. Verify artifact hashes and original-v2 raw-output reproduction for both checkpoints before attributing differences to the instruction. Changing the complete wrapper cannot isolate the Unknown clause alone. Keep strict scoring and the failed training gates unchanged; manually inspect units and wrong answers. The 40-answer diagnostic is implemented/local-tested but not GPU-executed. This does not select a new benchmark protocol or change model/training settings. See `docs/data/UTILITY_PROMPT_DIAGNOSTIC.md`.

## 2026-11-05 — Verified utility instruction sensitivity; no pilot promotion

The forty-answer diagnostic reproduces original-v2 outputs exactly and verifies all downloaded evidence. The arithmetic-specific wrapper removes base abstention and yields 7/10 strict, versus 0/10 under original-v2. Adapter arithmetic-v1 is 6/10 strict: its additional mismatch is correct `12 kilometers` rather than bare `12`. Both checkpoints make the same three numerical errors and are manually 7/10 numerically correct. Record sensitivity to the complete instruction on these probes, not proof that the Unknown clause alone caused it, broad arithmetic preservation, or numerical degradation from the one strict-score difference. Keep original v2 baselines/gates and failed pilot unchanged; alternative base remains below the original >=8/10 gate. Define any next utility protocol/probe prospectively before new SFT. No model, threshold or training change selected. Evidence: `experiments/runs/m1_utility_prompt_diagnostic_20261105/`.

## 2026-11-05 — Bounded thinking-mode feasibility check before more SFT

Prepare a development-only inference diagnostic using the same pinned base and saved adapter. Keep greedy reproduction controls; compare thinking on/off using identical sampling (temperature 0.6, top-p 0.95, top-k 20, min-p 0) and 1,024-token caps. Qwen's pinned official model card warns against greedy thinking, so changing thinking alone under greedy is not our primary comparison. Evaluate ten utility questions on both checkpoints and all 84 development questions on the adapter, for 312 predictions across three arms. Save visible generated reasoning separately, require a complete thinking boundary before scoring a thinking final answer, and record caps/failures without selective retries. This is one-seed, resource-bounded feasibility, not final evaluation, proof of internal composition, post-unlearning leakage or a replacement for the original failed gates. Adapter training used non-thinking mode; possible mismatch is part of diagnosis. Implemented/local-tested; no GPU results. See `docs/data/THINKING_MODE_DIAGNOSTIC.md`.

## 2026-11-05 — Verified thinking-mode tradeoff; original pilot still failed

The 312-prediction inference run verifies all original greedy controls. Under matched sampling, thinking changes arithmetic base 7/10 to 10/10 and adapter 6/10 to 10/10, while adapter fictional-fact development changes 73/84 to 21/84. Base thinking has three correct bare integers with recorded format violations; adapter thinking development has one capped/unclosed response and 62 additional completed final-answer failures. Inspected traces/finals show invented or mixed associations and some contradictions. Treat non-thinking-trained adapter versus thinking evaluation context as a plausible explanation, not proof of a causal mechanism, erased weights or broad incapacity. No weights were updated. Keep original gates/results unchanged, including the greedy R composite failure; do not select the sampled non-thinking 18/24 result as a replacement. Next design a separately versioned tiny training/evaluation retry aligned with intended mode(s) before scaling or M2. No specific new training hyperparameters or thresholds selected. Evidence: `experiments/runs/m1_thinking_mode_diagnostic_20261105/`.

## 2026-11-05 — Fresh-base mixed-mode tiny M1 retry v1

Keep the original pinned base, twelve-unit selection, LoRA/optimizer hyperparameters and 120 attempts. Train a fresh adapter, not the old diagnostic adapter, on 168 rows: each of 84 original training questions in answer-only form and thinking form with a deterministic canonical fact-chain rationale. Rationale supervision is constructed from train entities, not observed model thought; all assistant response/EOS tokens are supervised with prompts/padding masked and no truncation. Both fact modes must independently satisfy the original F/R learning thresholds; thinking arithmetic must have a measurable >=8/10 base, exact prior reference reproduction, <=10-point post drop, post format/cap/boundary checks, changed LoRA, 120 applied updates and exact fresh-base reload across all suites. Greedy non-thinking versus sampled thinking are separately declared go/no-go suites, not an isolated causal mode comparison. Same updates do not match token/compute exposure to the 84-row original; log actual per-mode exposure. These utility questions are development probes after repeated diagnosis, not final evidence of broad preservation. Implemented/local-tested/tokenizer-prepared, not GPU-trained; old pilot remains failed. See `docs/data/TINY_M1_MIXED_PILOT.md`.

## Explicitly NOT decided yet

- The final model checkpoint/revision, Python/library versions, remote provider and precision/quantization settings.
- Final dataset size, templates, exact entity splits, train/validation/test ratios and seeds.
- The numerical thresholds defining sufficient M1 learning, acceptable M2 forgetting and nontrivial recovery.
- Exact optimizer, learning rates, `alpha`, `beta`, RAFG thresholds/intervals/bursts and RASP `lambda`/module granularity.
- Whether either proposed method will outperform tuned, fair baselines; no gains have been measured.
- Whether both projects will remain separate proposals or one will become the final dissertation focus.

## Change policy

When a choice becomes final, move it to Confirmed with a date and reason. Record superseded decisions instead of erasing history. Add links to experiment configs/results supporting changes.
