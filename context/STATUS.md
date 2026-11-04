# Current Project Status

**As of:** 2026-11-04  
**Current phase:** Research definition and repository scaffolding — **before the first documented training run**.

> This status reflects the conversations and project-folder screenshot, not an inspection of running code. Nothing below should be marked complete without a saved artifact or logged experiment.

## What is already available

- [x] Two proposed project guides: `docs/proposals/RAFG.pdf` and `docs/proposals/RASP_Project_Guide.pdf` (visible in the provided folder screenshot).
- [x] Problem framed: persistence of targeted forgetting under later legitimate reasoning fine-tuning.
- [x] Shared pipeline agreed: M0 → M1 → M2 → undefended recovery pilot → separate RAFG/RASP branches.
- [x] Seven evaluation metrics identified; staged priorities agreed.
- [x] VS Code-centric, no-paid-API workflow discussed, with Colab/Kaggle as candidate compute backends.
- [x] Portable execution/storage architecture documented: GitHub for code and context, Kaggle as primary remote GPU, Colab as fallback, and private Hugging Face storage for durable model artifacts.
- [x] Private artifact repository created at `tharundevulapalli/persistent-llm-unlearning-checkpoints`; Kaggle Secrets authentication and a committed-version upload/download checksum round-trip were verified.
- [x] Reproducible M0 smoke-test runner and thin Kaggle/VS Code notebook implemented locally; syntax and notebook structure validated.
- [ ] Research code / experiments verified to run.
- [ ] Synthetic dataset generated and validated.
- [ ] M1 and M2 checkpoints produced.
- [ ] Recovery demonstrated and measured.
- [ ] RAFG or RASP method results collected.

## Stage gates

| Stage | Current status | Exit evidence |
|---|---|---|
| 0. Repository + environment | In progress: Git/context, private artifact round-trip and local smoke-test implementation established; remote GPU/model execution pending | Git baseline, environment/config file, remote GPU smoke test |
| 1. Synthetic data | Not started / unverified | Seeded generator, saved entity-level F/R/N/T splits, leakage checks |
| 2. M0 → M1 SFT | Not started / unverified | Reproducible M1 adapter and held-out direct/multi-hop learning metrics |
| 3. M1 → M2 unlearning | Not started / unverified | Saved canonical M2, reduced target accuracy, reasonable retain/reasoning accuracy |
| 4. Ordinary recovery pilot | Not started / unverified | From M2, logs across continuation steps for unrelated and similar-structure conditions |
| 5A. RAFG trials | Future | Trigger logging, recovery/utility/overhead vs fixed and periodic guards |
| 5B. RASP trials | Future | Module scores, recovery/utility/overhead vs uniform, random and forget-only controls |
| 6. Final comparison | Future | Same M2, matched tests/cost, seed variability, limitations |

## Immediate next actions (in order)

1. **Review the two proposal PDFs** and ensure paths/filenames are correct. Keep the PDFs unchanged as proposal sources.
2. **Initialize/verify Git and `.gitignore`.** Do not commit secrets, company information, large model files or disposable training outputs. The screenshot shows `U` beside files, which normally means untracked in VS Code.
3. **Run `experiments/notebooks/00_m0_smoke_test.ipynb` on a Kaggle GPU kernel from VS Code**: verify CUDA, package imports, the resolved Qwen3-0.6B revision, tokenizer/chat template, model loading and a short inference. Use Colab only as the planned fallback. The runner is implemented but has not been executed on Kaggle.
4. **Define data schema and deterministic splits** (`F`, `R`, `N`, `T`; and separate RAFG monitor/guard prompts later).
5. **Train on a tiny subset first** to prove SFT, prompt masking, save/load and direct/multi-hop evaluation.
6. Scale to a first reproducible **M1**, then implement and validate **M2**.
7. Only after M2 is validated, run **ordinary post-unlearning recovery pilot**. Branch into RAFG/RASP after the pilot.

## Known risks and checks

- **M1 fails to learn:** check question/answer formatting, chat template, label masking, learning rate, and train/held-out results.
- **M2 fails to forget or loses utility:** check Gradient Difference sign and coefficient balance, retain sampling and checkpoint lineage.
- **No recovery after later SFT:** this is a possible research outcome; test both preplanned continuation conditions and report honestly.
- **Weak small-model reasoning:** use controlled one- to three-hop tasks as primary reasoning evaluation; harder datasets are optional.
- **Cloud notebook resets:** persist adapters, data manifests and results outside ephemeral runtimes at every milestone.
- **Office laptop handoff:** perform a clean restore test on the personal laptop before January 29; comply with employer device policy.

## Reporting rules

Do not write percentages, plots, speed claims or method wins into `STATUS.md` until they are produced by logged experiments. When runs exist, link each claim to its config, checkpoint and results file.
