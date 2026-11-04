# Current Project Status

**As of:** 2026-11-05

**Current phase:** Both M0 audits preserved and verified. V2 baseline is 1,259 abstentions plus one incorrect answer, with zero format/cap issues. Remaining candidate acceptance review and tiny M1 learning criteria precede training.

> This status distinguishes documented plans from inspected artifacts and logged experiments. Nothing is marked complete without saved evidence.

## What is already available

- [x] Two proposed project guides: `docs/proposals/RAFG.pdf` and `docs/proposals/RASP_Project_Guide.pdf` (visible in the provided folder screenshot).
- [x] Problem framed: persistence of targeted forgetting under later legitimate reasoning fine-tuning.
- [x] Shared pipeline agreed: M0 → M1 → M2 → undefended recovery pilot → separate RAFG/RASP branches.
- [x] Seven evaluation metrics identified; staged priorities agreed.
- [x] VS Code-centric, no-paid-API workflow discussed, with Colab/Kaggle as candidate compute backends.
- [x] Portable execution/storage architecture documented: GitHub for code and context, Kaggle as primary remote GPU, Colab as fallback, and private Hugging Face storage for durable model artifacts.
- [x] Private artifact repository created at `tharundevulapalli/persistent-llm-unlearning-checkpoints`; Kaggle Secrets authentication and a committed-version upload/download checksum round-trip were verified.
- [x] Reproducible M0 smoke-test runner and thin Kaggle/VS Code notebook implemented locally; syntax and notebook structure validated.
- [x] Interactive M0 smoke test passed through VS Code on a Kaggle T4 x2 runtime: CUDA, immutable model revision resolution, tokenizer/chat template, BF16 model loading and bounded inference were verified.
- [x] Fresh Kaggle Save & Run All version reproduced the pinned M0 smoke test from Git commit `bb50b38e2315626d1e2eae3dc719f85a5ead8561`.
- [x] Synthetic relational dataset schema and deterministic split/validation plan documented in `docs/data/DATASET_SPEC.md`; no accepted full pilot bundle generated yet.
- [x] Standard-library dataset generator, validators and reproducibility tests implemented; two independent tiny bundles matched byte-for-byte with zero validation errors.
- [x] Tiny Qwen tokenizer-budget audit passed: 636 `N_rel` vs 648 `N_reason` tokens (1.89% difference, within 5%); automatic near-duplicate checks found no pairs at similarity ≥ 0.92.
- [x] Tiny manual review passed: exact public searches found no collisions for the 9 sampled authors/books/prizes, prompt wording was coherent, and final evaluation was expanded to two exclusive paraphrases per task (126 tiny final prompts).
- [x] M0 infrastructure and smoke-test code verified on a fresh Kaggle committed run.
- [x] Dataset-generation and validation code implemented and verified in tiny mode.
- [x] Full 30/90/60 candidate saved at `data/candidates/pilot-v1/seed-42/`; two independent generations matched byte-for-byte and repository copy matched both.
- [x] Full automated validation passed with zero errors/near-duplicate pairs; tokenizer audit passed at 12,480 N_rel versus 12,840 N_reason tokens (2.88% difference). Nine unit tests pass.
- [x] M0 knowledge-audit runner, fixed config and VS Code/Kaggle notebook prepared; local preflight verifies 1,260 non-final prompts covering 180 units. All 13 local tests pass.
- [x] User-reported Kaggle M0 audit completed from Git commit `a774aa00b0b13ebdb748727abc741ecd86123401`: 1,260 predictions, 0 exact matches, 0 empty final responses, and 2 longer-response year mentions requiring review. Summary preserved in `experiments/runs/m0_knowledge_audit_20261105/reported_summary.json`.
- [x] Both pasted year flags reviewed against candidate records: one year embedded in an incorrect book/prize claim; one semantically correct year in prose. Preserve zero strict exact matches and record the prose answer separately in `experiments/runs/m0_knowledge_audit_20261105/match_review.json`.
- [x] All five original audit files downloaded and saved under `experiments/runs/m0_knowledge_audit_20261105/`; raw checksum and all 1,260 inputs/summary counts verified. A 42-response nonmatch sample across all 21 partition/task groups was reviewed; evidence is in `artifact_verification.json`.
- [x] Defined and implemented `answer-only-v2`: shared `Answer:` instruction/parser, 128-token cap, per-response EOS/cap logging, separate format/answer/mention counts and compatible future SFT target renderer. All 23 unit tests pass; pinned-tokenizer prompt rendering and 1,260-question CPU preflight pass.
- [x] User-reported v2 Kaggle audit completed from Git commit `b2df70757ed57b53270913ecd453d5fc36a7ac12`: 1,260 predictions, zero parsed-answer matches, zero format violations and zero token-limit hits. Concise summary saved in `experiments/runs/m0_answer_only_audit_20261105/reported_summary.json`.
- [x] All five original v2 files preserved; prediction/config/source/input hashes verified and all 1,260 scores/group counts recomputed. Raw distribution: 1,259 `Unknown` responses and one wrong book answer; all rows log EOS, generated counts 5–18. Inspected 42 samples across 21 groups plus the one exception. Evidence: `experiments/runs/m0_answer_only_audit_20261105/artifact_verification.json`.
- [ ] Training code implemented and verified.
- [ ] Full synthetic candidate accepted after manual collision review and M0 accidental-knowledge audit.
- [ ] M1 and M2 checkpoints produced.
- [ ] Recovery demonstrated and measured.
- [ ] RAFG or RASP method results collected.

## Stage gates

| Stage | Current status | Exit evidence |
|---|---|---|
| 0. Repository + environment | **Complete:** Git baseline, private artifact round-trip, pinned environment, interactive GPU test and committed M0 rerun verified | Git baseline, environment/config file, remote GPU smoke test |
| 1. Synthetic data | Full automated gates, original/v2 audit verification and response sampling passed; remaining manual collision/acceptance review and pilot criteria precede training | `experiments/runs/m0_answer_only_audit_20261105/`, protocol, candidate data and original evidence |
| 2. M0 → M1 SFT | Not started / unverified | Reproducible M1 adapter and held-out direct/multi-hop learning metrics |
| 3. M1 → M2 unlearning | Not started / unverified | Saved canonical M2, reduced target accuracy, reasonable retain/reasoning accuracy |
| 4. Ordinary recovery pilot | Not started / unverified | From M2, logs across continuation steps for unrelated and similar-structure conditions |
| 5A. RAFG trials | Future | Trigger logging, recovery/utility/overhead vs fixed and periodic guards |
| 5B. RASP trials | Future | Module scores, recovery/utility/overhead vs uniform, random and forget-only controls |
| 6. Final comparison | Future | Same M2, matched tests/cost, seed variability, limitations |

## Immediate next actions (in order)

1. Finish the remaining manual collision review and record candidate acceptance without changing examples to force zero M0 accuracy. Both baselines are preserved; do not rerun them merely because most responses abstain.
2. Define tiny M1 learning/utility criteria and implement the SFT renderer, answer-token masking and save/reload checks using the same v2 evaluation wrapper.
3. **Train on a tiny subset first** to prove SFT, prompt masking, save/load and direct/multi-hop evaluation.
4. Scale to a first reproducible **M1**, then implement and validate **M2**.
5. Only after M2 is validated, run **ordinary post-unlearning recovery pilot**. Branch into RAFG/RASP after the pilot.

## Known risks and checks

- **M1 fails to learn:** check question/answer formatting, chat template, label masking, learning rate, and train/held-out results.
- **M2 fails to forget or loses utility:** check Gradient Difference sign and coefficient balance, retain sampling and checkpoint lineage.
- **No recovery after later SFT:** this is a possible research outcome; test both preplanned continuation conditions and report honestly.
- **Weak small-model reasoning:** use controlled one- to three-hop tasks as primary reasoning evaluation; harder datasets are optional.
- **Cloud notebook resets:** persist adapters, data manifests and results outside ephemeral runtimes at every milestone.
- **Office laptop handoff:** perform a clean restore test on the personal laptop before January 29; comply with employer device policy.

## Reporting rules

Do not write percentages, plots, speed claims or method wins into `STATUS.md` until they are produced by logged experiments. When runs exist, link each claim to its config, checkpoint and results file.
