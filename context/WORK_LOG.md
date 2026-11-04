# Work Log

Chronological notes of decisions, experiments and blockers. **Planning activities are not training results.** Add a new dated entry after each meaningful work session.

## 2026-11-04 — Project context consolidation

**Activity:** Prepared the repository-context drafts from the RAFG and RASP proposal guides and the project conversations.

**Established in discussion:**
- Two independent candidate safeguards for the same benign-relearning problem: RAFG (reactive) and RASP (preventive).
- One shared pipeline through trained checkpoint M1, unlearned checkpoint M2 and an undefended recovery pilot.
- Both method branches should start from the same canonical M2, not separately retrained/unlearned versions.
- Evaluation plan includes seven metrics: direct and multi-hop forget accuracy, trace leakage when visible, recovery, retain accuracy, new-task reasoning accuracy and computational overhead.
- Preferred workflow: VS Code for development, free cloud GPU for training where available; portable Git-backed code and persistent checkpoint backups.

**Artifacts:** `docs/proposals/RAFG.pdf`, `docs/proposals/RASP_Project_Guide.pdf`; Markdown context drafts for this repository.

**Not established yet:** No verified synthetic dataset, M1/M2 adapters, recovery results, RAFG/RASP experiments, GPU benchmarks or measured method improvements were provided.

**Next step:** Check Git/environment, choose the first remote execution setup, and run a small model load/inference smoke test.

---

## 2026-11-04 — Portable execution and artifact architecture

**Goal:** Define how development, remote GPU execution and persistent storage continue across the company Mac and personal Dell.

**Work completed:** Added a directional architecture diagram, storage ownership table, run lifecycle and credential boundaries to `README.md`. Recorded Kaggle as the primary remote GPU environment, Colab as fallback, GitHub as the code/context system of record and a private Hugging Face model repository as durable M1/M2 adapter/checkpoint storage.

**Evidence:** Documentation changes only; no runtime connection, artifact upload or model restore has been tested yet.

**Decision and next step:** Use the documented architecture for the first Kaggle-from-VS-Code smoke test. Before M1/M2 work, verify that a small test artifact can be uploaded to private storage and restored in a fresh runtime.

**Cost / time (if measured):** Not measured; no compute run.

---

## 2026-11-04 — Private artifact storage round-trip

**Goal:** Verify that Kaggle can securely persist and restore project artifacts through the private Hugging Face model repository.

**Environment / model:** Kaggle CPU notebook; no accelerator and no model loaded.

**Input / configuration:** Repository `tharundevulapalli/persistent-llm-unlearning-checkpoints`; repository-scoped Hugging Face token stored as the attached Kaggle secret `HF_TOKEN`.

**Work completed:** Created the private model repository, configured a fine-grained token for the selected repository, attached it through Kaggle Secrets, authenticated from a Kaggle notebook, uploaded `storage-tests/kaggle-roundtrip.txt`, downloaded it and compared SHA-256 checksums.

**Evidence:** The committed Kaggle notebook version output confirmed authentication as `tharundevulapalli`, successful upload and restore, and a matching checksum. The upload produced Hugging Face commit `8748fe1ff0eea0f45e79a47a990d96ae9c6bc086`. The token value was not shared or recorded; the checksum value was not copied into this work log.

**Observations / failures:** No failure was reported. Kaggle Version #1 reran the notebook independently and completed the round-trip. The first real adapter must receive its own restore and checksum verification.

**Decision and next step:** Keep the private Hugging Face model repository as the canonical adapter/checkpoint store. Next, connect VS Code to a Kaggle GPU runtime and perform the M0 CUDA, dependency, tokenizer, model-load and short-inference smoke test.

**Cost / time (if measured):** Not measured; CPU storage test only.

---

## 2026-11-04 — M0 smoke-test implementation

**Goal:** Prepare a reproducible, inference-only gate for the candidate Qwen3-0.6B base model before dataset generation or training.

**Environment / model:** Local code validation on the company Mac; target execution is a Kaggle CUDA runtime using `Qwen/Qwen3-0.6B`.

**Input / configuration:** Default seed 42, non-thinking chat-template mode, bounded 64-token response, Qwen-recommended non-thinking sampling settings. The requested Hub revision is resolved to an immutable commit before loading.

**Work completed:** Added `src/m0_smoke_test.py`, `requirements/m0-smoke.txt` and thin runner `experiments/notebooks/00_m0_smoke_test.ipynb`. The runner records the Git commit, resolved model revision, tokenizer/model classes, package versions, full environment freeze, CUDA/GPU details, peak allocated memory, timing, prompt and response. It performs no training or artifact upload.

**Evidence:** `python3 -m py_compile src/m0_smoke_test.py`, CLI help execution, `jq empty experiments/notebooks/00_m0_smoke_test.ipynb` and `git diff --check` passed locally.

**Observations / failures:** The model and tokenizer were not downloaded or executed locally. CUDA behavior, Kaggle compatibility, memory use and inference output remain unverified.

**Decision and next step:** Commit/push the runner, connect the local notebook to a Kaggle GPU kernel, execute it, review `result.json`, then run a committed Kaggle version before marking M0 complete.

**Cost / time (if measured):** No GPU compute used.

---

## 2026-11-04 — Interactive M0 GPU/model smoke test

**Goal:** Verify the candidate base model and reproducible loader on remote CUDA compute before dataset generation or training.

**Environment / model:** VS Code notebook connected to a Kaggle Jupyter runtime with two Tesla T4 GPUs; Python 3.13.16; PyTorch 2.11.0+cu130; CUDA 13.0; `Qwen/Qwen3-0.6B` resolved to Hub commit `c1899de289a04d12100db370d81485cdf75e47ca`.

**Input / configuration:** Repository commit `d2e9937d1222c0651ba0ecbf8a42096a9f059ae9`; seed 42; non-thinking chat template; sampling with temperature 0.7, top-p 0.8 and top-k 20; maximum 64 new tokens.

**Work completed:** Connected VS Code to the Kaggle GPU kernel, cloned the exact Git repository state, installed the M0 requirements, resolved the immutable model revision, loaded the tokenizer/model in BF16 and ran one bounded inference. Captured and pinned direct dependency versions. Updated the runner from deprecated `torch_dtype` to `dtype` for the verified Transformers version.

**Evidence:** `experiments/runs/m0_smoke_20261104/result.json` records success, response `The model loaded successfully.`, 15.1902-second load, 2.548-second generation, six generated tokens and 1.311 GiB peak allocated GPU memory. Verified direct packages: Accelerate 1.15.0, huggingface-hub 1.33.0, safetensors 0.8.0 and Transformers 5.20.2.

**Observations / failures:** Public Hub access emitted an unauthenticated-rate-limit warning but completed successfully. Transformers emitted a `torch_dtype` deprecation warning; the local runner is now corrected. The remote full `environment.freeze.txt` has not yet been copied into Git. This was an interactive run, not a committed Kaggle version.

**Decision and next step:** Treat the candidate model/revision as smoke-tested, not yet as a final benchmark choice. Push the pinned requirements and loader fix, then execute a committed Kaggle notebook version and preserve its output before closing M0.

**Cost / time (if measured):** Load and generation timings recorded above; no monetary cost measured.

---

## 2026-11-04 — Committed M0 reproducibility run

**Goal:** Reproduce the pinned M0 smoke test from a clean Kaggle Save & Run All environment and close the M0 stage gate.

**Environment / model:** Fresh committed Kaggle run with two Tesla T4 GPUs; Python 3.13.16; PyTorch 2.11.0+cu130; CUDA 13.0; Transformers 5.20.2; `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.

**Input / configuration:** Git commit `bb50b38e2315626d1e2eae3dc719f85a5ead8561`; pinned `requirements/m0-smoke.txt`; seed 42; non-thinking mode; temperature 0.7, top-p 0.8, top-k 20; maximum 64 new tokens.

**Work completed:** Imported the Git-tracked notebook into Kaggle and executed Save & Run All from a clean state. The run cloned the expected Git commit, installed pinned dependencies, loaded the immutable model revision, ran inference, validated the structured result and generated both `result.json` and `environment.freeze.txt` in Kaggle output.

**Evidence:** The supplied Kaggle log records `success: true`, response `The model loaded successfully.`, 12.8933-second load, 2.6543-second generation, six generated tokens, 1.311 GiB peak allocated GPU memory and matching runtime/package versions. The structured log result is preserved as `experiments/runs/m0_smoke_20261104/committed_result.json`.

**Observations / failures:** No test failure occurred. Benign warnings concerned frozen debugger modules, unauthenticated public-Hub rate limits and syntax warnings inside Kaggle's post-run nbconvert/mistune tooling. The prior `torch_dtype` deprecation warning did not recur.

**Decision and next step:** Mark the M0 environment/model-loading gate complete. The next task is to design and review deterministic entity-first schemas and F/R/N/T splits before implementing full dataset generation. No training has occurred.

**Cost / time (if measured):** Committed notebook log reached result generation in about 62 seconds and post-processing completed in about 73 seconds; no monetary cost measured.

---

## 2026-11-04 — Synthetic dataset schema and split design

**Goal:** Define a reproducible, leakage-resistant relational dataset and evaluation boundary before implementing generation or beginning M1 training.

**Environment / model:** Documentation-only local work; no model, dataset generator or compute runtime used.

**Input / configuration:** Proposal-derived pilot of 120 M1 knowledge units with 30 forget and 90 retain units; pilot seed 42; two planned post-M2 conditions (unrelated reasoning and structurally similar new relations).

**Work completed:** Added `docs/data/DATASET_SPEC.md` defining the author → book → prize → award-year graph, canonical record/prompt schemas, stable SHA-256 entity splitting, F/R/N roles, held-out T boundary, direct/2-hop/3-hop tasks, prompt-use partitions for M1/M2/RAFG/RASP, leakage and uniqueness checks, recovery eligibility, manifests, planned file layout and a pre-training acceptance gate.

**Evidence:** Documentation links and context were updated. `T` is explicitly treated as a held-out prompt view rather than a trainable entity partition. No synthetic records or experimental metrics were created.

**Observations / failures:** The design leaves final `N` budgets, per-partition template counts, name-generation constraints, metric thresholds and final seed list pending. The provisional `N_rel` default is 60 units and must be reviewed before implementation.

**Decision and next step:** Review and resolve the pending pilot choices, record them in `DECISIONS.md`, then implement the generator and validators on CPU. Full generation and M1 training remain unauthorized until the dataset acceptance gate passes.

**Cost / time (if measured):** No compute used.

---

## 2026-11-04 — Tiny synthetic generator and validation implementation

**Goal:** Implement and verify deterministic dataset generation without creating the accepted full pilot or starting training.

**Environment / model:** Local CPU; Python standard library only; no model or remote compute used.

**Input / configuration:** `configs/data/pilot_v1.json` with master seed 42 and full planned counts; `--tiny` overrides counts to 2 F, 4 R and 3 N_rel units for implementation tests.

**Work completed:** Added deterministic SHA-256-ranked entity generation, entity-first F/R splitting, disjoint N_rel generation, 11 exclusive natural-language templates per relational task, direct/2-hop/3-hop prompt generation, row-matched arithmetic N_reason prompts, prompt-use partitions, leakage/uniqueness/path validation, manifests, per-file checksums, manual-review samples, CLI generation/validation commands and standard-library unit tests.

**Evidence:** Six unit tests passed. Two independent CLI tiny generations were byte-for-byte identical. Each contained 2 F, 4 R and 3 N_rel units; validation reported zero errors, checksum verification passed, and prompt counts matched expectations including 21 N_rel and 21 N_reason rows plus 63 held-out final prompts. A dedicated test confirms forget updates target only the direct author–book relation.

**Observations / failures:** Manual inspection showed coherent fictional examples and correct graph paths. Validation intentionally retains two warnings: N_reason/N_rel are row-matched rather than tokenizer-budget matched, and near-duplicate/real-name collision review remains manual. The full 30/90/60 dataset was not generated.

**Decision and next step:** Review the warnings and template breadth, then authorize two clean full generations, compare all bytes/checksums, perform manual review and run the M0 accidental-knowledge audit before committing an accepted dataset.

**Cost / time (if measured):** Final unit-test run completed in 0.123 seconds; no GPU or paid compute used.

---

## 2026-11-04 — Tiny tokenizer-budget and near-duplicate audits

**Goal:** Resolve the automated validation warnings before any full-pilot generation.

**Environment / model:** Temporary `/tmp` Python virtual environment on CPU; pinned `Qwen/Qwen3-0.6B` tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`; no PyTorch or model weights loaded.

**Input / configuration:** Tiny bundle with 2 F, 4 R and 3 N_rel units; 21 N_rel and 21 N_reason training rows; non-thinking chat template; 5% token-budget tolerance; normalized-surface similarity rejection threshold 0.92.

**Work completed:** Added exact tokenizer configuration, a tokenizer-only audit CLI, pinned audit requirements, automatic near-duplicate rejection/reporting, two additional unit tests and a tracked tiny-validation summary. Corrected the audit to read `input_ids` from the tokenizer mapping after the first real run exposed an invalid two-token count. Shortened the N_reason instruction by one token per row after a valid first comparison missed tolerance by 0.19 percentage points.

**Evidence:** Eight unit tests passed. Final audit measured N_rel at 636 total tokens and N_reason at 648, a 12-token / 1.886792% difference within the 5% tolerance, with equal 21-row update opportunities. No author, book or prize near-duplicate pairs were present at similarity ≥ 0.92. Evidence is summarized in `experiments/runs/data_tiny_validation_20261104/summary.json`.

**Observations / failures:** The initial real audit incorrectly counted mapping keys and was rejected; the corrected audit then measured a 5.188679% mismatch and failed as designed. After the prompt adjustment, it passed. Automatic similarity checks do not prove names are absent from the real world; that remains a manual review item.

**Decision and next step:** Preserve equal rows/steps and require ≤5% token difference. Next review tiny surfaces/templates manually, then generate the full bundle twice and run the same gates before an M0 accidental-knowledge audit.

**Cost / time (if measured):** CPU-only; no GPU or paid compute used.

---

## 2026-11-05 — Tiny manual collision and template review

**Goal:** Complete the human-review portion of the tiny-data gate before authorizing full-pilot generation.

**Environment / model:** Local CPU generation plus exact-phrase public web searches; no model weights or training.

**Input / configuration:** Regenerated seed-42 tiny bundle containing 2 F, 4 R and 3 N_rel units; 9 authors, 9 books and 9 prizes; all prompt partitions and task families.

**Work completed:** Inspected representative questions/answers across every prompt partition and task. Exact searches for all tiny author names, book titles and prize names returned no results. Expanded each task family from 11 to 12 natural-language variants so every non-final partition keeps one exclusive template and final evaluation receives two exclusive held-out paraphrases.

**Evidence:** Eight tests still pass; validation reports zero errors and zero near-duplicate pairs; checksums pass; final-test count increased from 63 to 126 without changing N_rel/N_reason training rows. The pinned tokenizer audit remains unchanged and passes at 636 versus 648 tokens (1.89%).

**Observations / failures:** Some prompts are deliberately concise, but all inspected questions are grammatical and resolve to the intended unique answer. Exact public search is evidence against obvious collisions, not proof that a synthetic surface has never appeared anywhere.

**Decision and next step:** Accept two final paraphrases per task for the pilot, subject to reassessment after tiny M1. The next gated task is two independent full 30/90/60 generations, byte/checksum comparison and full tokenizer/near-duplicate validation; the accepted bundle must still wait for M0 accidental-knowledge testing.

**Cost / time (if measured):** No GPU or paid compute used.

---

## 2026-11-05 — Full synthetic candidate generation and automated validation

**Goal:** Generate the full 30 F / 90 R / 60 N_rel candidate and verify reproducibility and full-size automated gates before the M0 knowledge audit.

**Work completed:** The first generation rejected duplicate arithmetic questions. Corrected the arithmetic generator to skip duplicates deterministically and added a 420-row uniqueness/correctness regression test. Added complete config/config hash and generator-source hashes to manifests. Generated the corrected candidate twice in separate processes with Python hash seeds 1 and 97 and saved a matching repository copy at `data/candidates/pilot-v1/seed-42/`.

**Evidence:** Nine tests pass. Both independent outputs and the repository copy match byte-for-byte. Checksums pass, validation reports zero errors and zero near-duplicate pairs. The full pinned-tokenizer audit measured 12,480 N_rel tokens and 12,840 N_reason tokens across 420 rows each (2.884615%, within the configured 5% limit). M1 has 840 training prompts; final evaluation has 2,520 prompts. See `experiments/runs/data_full_validation_20261105/summary.json` and `token_budget.json`.

**Provenance:** Baseline Git commit `18dd6b9a06bfc71bae6b30272d1900abc87661dd` plus exact source hashes in the manifest identify the uncommitted duplicate correction/provenance additions. The candidate checksum-index digest is `1a70624618acb6a849f1605766a79eef0f66f8b0aab938ed500152d1705ebe05`. The manifest's timestamp is fixed config metadata; this entry records the actual generation date.

**Manual review:** Spot-checked the first and last two units from each of F, R and N_rel (12 records). Full external collision review and M0 accidental-knowledge testing are still pending. Equal text volume/rows is a preliminary control; trainer step budgets and answer masking remain to be defined.

**Files changed:** Generator, regression tests, candidate dataset bundle, full validation reports, `data/README.md`, root README, dataset specification, status, decision log and this work log.

**Next step:** Review the full candidate and run the pinned base model on Kaggle to audit accidental knowledge, then record a dataset acceptance decision before starting the tiny M1 learning pilot.

**Compute:** CPU and cached tokenizer only; no model weights loaded, GPU compute or training used. Changes remain local for user review and push.

---

## 2026-11-05 — Prepare M0 knowledge-audit runner

**Goal:** Prepare reproducible baseline predictions on the full candidate before M1 fact learning.

**Files changed:** Added `src/m0_knowledge_audit.py`, `configs/evaluation/m0_knowledge_audit.json`, `experiments/notebooks/01_m0_knowledge_audit.ipynb`, `tests/test_knowledge_audit.py` and `docs/data/M0_KNOWLEDGE_AUDIT.md`; updated README, dataset acceptance instructions, decisions and status. Previous local full-generation changes are preserved.

**Implemented:** Candidate checksum verification, immutable model revision, coverage checks, batched non-thinking greedy inference on one Kaggle GPU, native-capability-aware dtype selection (FP16 on T4), per-prediction raw outputs, exact/mention review flags, separate year-match counts, output overwrite protection, batch-level progress/prediction flushes and runtime/config/hash metadata. The notebook clones a fresh Git working copy and prints its commit before execution.

**Evidence:** Thirteen unit tests pass, including boundary-aware match scoring, visible-thought exclusion, year separation, all-unit non-final input coverage, modified-checksum rejection and final-test input rejection. All five notebook code cells compile. A CPU-only `--plan-only` run verified 1,260 prompts covering F=210, R=630 and N_rel=420 across 180 units. No model weights were loaded; no predictions or model metrics were produced.

**Protocol:** Use development/non-final prompts. Preserve matches rather than automatically regenerating examples to force zero M0 accuracy. Year matches may be chance. Dataset acceptance requires manual interpretation plus the remaining full collision review.

**Next step:** User commits/pushes local files, connects the new notebook to a Kaggle GPU kernel, executes it and preserves the printed output directory. Review actual outputs before marking the audit complete or accepting the dataset.

---

## Work-log entry template (copy for future sessions)

### YYYY-MM-DD — Short description

**Goal:**

**Environment / model:** (device, CUDA/MPS, base revision, library versions)

**Input / configuration:** (data split, seed, config path)

**Work completed:**

**Evidence:** (checkpoint path, metrics file, log, commit)

**Observations / failures:**

**Decision and next step:**

**Cost / time (if measured):**
