# Current Project Status

**As of:** 2026-11-05

**Current phase:** Full synthetic pilot candidate generated and automatically validated; manual collision review and M0 knowledge audit pending before the first training run.

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
- [x] M0 knowledge-audit runner, fixed config and VS Code/Kaggle notebook prepared; local preflight verifies 1,260 non-final prompts covering 180 units. All 13 local tests pass. GPU audit execution remains pending.
- [ ] Training code implemented and verified.
- [ ] Full synthetic candidate accepted after manual collision review and M0 accidental-knowledge audit.
- [ ] M1 and M2 checkpoints produced.
- [ ] Recovery demonstrated and measured.
- [ ] RAFG or RASP method results collected.

## Stage gates

| Stage | Current status | Exit evidence |
|---|---|---|
| 0. Repository + environment | **Complete:** Git baseline, private artifact round-trip, pinned environment, interactive GPU test and committed M0 rerun verified | Git baseline, environment/config file, remote GPU smoke test |
| 1. Synthetic data | Full candidate generated; independent byte comparison, checksums, automated validation and tokenizer audit passed; full manual collision review and M0 knowledge audit pending | `data/candidates/pilot-v1/seed-42/` and `experiments/runs/data_full_validation_20261105/` |
| 2. M0 → M1 SFT | Not started / unverified | Reproducible M1 adapter and held-out direct/multi-hop learning metrics |
| 3. M1 → M2 unlearning | Not started / unverified | Saved canonical M2, reduced target accuracy, reasonable retain/reasoning accuracy |
| 4. Ordinary recovery pilot | Not started / unverified | From M2, logs across continuation steps for unrelated and similar-structure conditions |
| 5A. RAFG trials | Future | Trigger logging, recovery/utility/overhead vs fixed and periodic guards |
| 5B. RASP trials | Future | Module scores, recovery/utility/overhead vs uniform, random and forget-only controls |
| 6. Final comparison | Future | Same M2, matched tests/cost, seed variability, limitations |

## Immediate next actions (in order)

1. Commit/push the local candidate and audit files, then execute `experiments/notebooks/01_m0_knowledge_audit.ipynb` on a Kaggle GPU kernel from VS Code. Preserve all predictions and review matches according to `docs/data/M0_KNOWLEDGE_AUDIT.md`; full manual name/title review remains pending.
2. Record the acceptance decision and define M1 learning criteria before implementing a tiny training pilot.
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
