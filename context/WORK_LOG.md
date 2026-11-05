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

## 2026-11-05 — Reported Kaggle M0 knowledge-audit completion

**Activity:** User executed the audit notebook through a remote Kaggle kernel and supplied its completed summary. Saved a faithful structured transcription in `experiments/runs/m0_knowledge_audit_20261105/reported_summary.json` with an evidence/provenance README. Preserved the user's executed local notebook changes.

**Reported configuration:** Source commit `a774aa00b0b13ebdb748727abc741ecd86123401`; pinned Qwen3-0.6B revision; non-thinking greedy decoding; 32-token cap; FP16 on Tesla T4. Preflight verified the configured candidate checksums and 1,260 prompts across all 180 units.

**Reported observations:** Execution complete; 1,260 predictions; 0 normalized exact matches; 0 empty final responses; 2 expected-answer mentions, both year answers on author-to-year prompts (one R, one N_rel). Duration 117.4098 seconds and peak allocated memory 1.2503 GiB. Run timestamps are November 4 UTC / November 5 IST.

**Evidence limits:** Summary fields are internally consistent: group counts sum to 1,260 and mentions sum to 2. Full raw predictions and the two flagged responses have not yet been inspected locally. The supplied prediction checksum is recorded, not independently verified. Zero exact matches does not establish absence of knowledge.

**Files changed:** Reported-summary JSON, run README, root README, data README, audit guide, status, decisions and this work log. No generator, candidate records, model or training changes.

**Next step:** Inspect `matches_for_review.json`, preserve all five original output files from `/kaggle/working/m0-knowledge-audit-feenm09g`, verify the prediction checksum and sample nonmatches before dataset acceptance. Full manual collision review remains pending; training has not started.

---

## 2026-11-05 — Review two M0 year-mention flags

**Evidence:** User supplied the two complete flagged records. Preserved them in `experiments/runs/m0_knowledge_audit_20261105/reported_matches_for_review.json` and compared against local canonical entity records. Added `match_review.json` with separate interpretations.

**Findings:** For R record `ku_000080` (Elira Sable), the response mentions expected year 2019 but supplies a book/prize inconsistent with *The Wandering Delta* / *Young Elm Prize*. For N_rel record `nrel_000009` (Zella Pryce), the response states expected year 2023 correctly in prose. The latter is semantically correct despite failing the unchanged whole-string exact-match metric. Neither response alone establishes pre-existing knowledge of the synthetic graph.

**Decision:** Preserve the reported zero exact matches, raw evidence and a separate manual year-answer finding. No candidate changes/exclusions. Before subsequent model comparisons, specify consistent response-format handling rather than treating all prose answers as wrong.

**Files changed:** Added reported match JSON and review JSON; updated run README, status, decision log and work log. No scoring-code, model or candidate changes. The user's executed notebook is preserved.

**Next step:** Archive and download the original five audit output files, verify the raw-prediction checksum, inspect representative nonmatches, and complete the remaining collision review. Training remains unstarted.

---

## 2026-11-05 — Preserve and independently verify original audit files

**Work completed:** Located the user-downloaded `m0-knowledge-audit-feenm09g-results.zip`, checked ZIP integrity and expected five entries, and extracted the original files without overwrite into `experiments/runs/m0_knowledge_audit_20261105/`. Preserved earlier pasted-summary/match transcriptions and the executed notebook.

**Verification:** Prediction SHA-256 matches `c5bc23c7ce65bf86b46deca5f84f90bcbf90146177c59577cea5ca2612fa80b8`. All 1,260 example IDs are unique, every original input field agrees with the saved candidate, config/input hashes agree, rescoring reproduces every summary group/count, and original flags match the two pasted records. The supplied summary transcription agrees with the original. Full checks and review samples are recorded in `artifact_verification.json`.

**Manual sample:** Reviewed the first and middle nonflagged row in each of 21 partition/task groups (42 responses). Observed invented associations, unrelated answers, refusals and several mid-sentence endings. The 32-token original audit did not log generated-token counts or EOS/cap termination, so an exact truncation rate cannot be determined. Preserve this baseline and define answer-format/output-length/termination controls before M1/M2 comparisons.

**Files changed:** Added original `plan.json`, `summary.json`, `predictions.jsonl`, `matches_for_review.json`, `environment.freeze.txt` and `artifact_verification.json` to the run folder; updated its README, root README, data README, audit guide, status, decision log and this work log. No candidate records, scoring code or model weights changed.

**Next step:** Stop the no-longer-needed Kaggle runtime, review/push local records, then resolve evaluation-protocol limitations and remaining manual collision review before dataset acceptance and the tiny M1 pilot. No training has occurred.

---

## 2026-11-05 — Implement versioned answer-only evaluation protocol

**Goal:** Address output-format and truncation limitations found in the original M0 audit before M1 training or final-test evaluation.

**Work completed:** Added `src/evaluation/answers.py` with a gold-independent answer parser, shared system instruction, future SFT answer renderer, EOS/padding-aware token accounting and protocol diagnostics. Added opt-in v2 support to the existing audit runner, a fixed v2 config, clean `02_m0_answer_only_audit.ipynb`, focused tests and `docs/data/EVALUATION_PROTOCOL.md`. Updated README, audit guide, status and decisions. The old config, scorer, candidate files and original audit outputs remain preserved.

**Protocol:** Ask for `Answer: <value>`; accept exact normalized answer fields (or a bare exact value with a recorded format violation), never count mere mentions/negations/alternatives as automatic correct answers. Keep prose semantic findings as separate manual evidence. Set a 128-token cap; log generated counts, first EOS and token-limit hits per response. Use `summary.answer_scoring` for v2 accuracy and retain the old whole-response diagnostic separately.

**Evidence:** All 23 unit tests pass, including original 1,260-row baseline-score preservation, invalid/ambiguous answer cases, EOS versus shared padding, EOS at the cap and aggregation. Six clean notebook code cells compile. The pinned cached Qwen tokenizer rendered the system instruction and non-thinking generation prompt successfully. CPU `--plan-only` verified unchanged candidate hashes, all 180 units and 1,260 inputs. One empty-label parsing bug was caught and corrected during testing.

**Limits:** No new model inference or training occurred. The revised baseline needs a Kaggle run and inspection before comparisons. Trainer token boundaries, loss masking, rendered token budgets and M1 learning thresholds still need implementation/definition; full manual collision review remains pending.

**Next step:** User reviews/pushes these files and runs the v2 notebook on a Kaggle GPU kernel, preserving its artifacts separately. Do not replace the original run or infer v2 model metrics from local parser tests.

---

## 2026-11-05 — Reported v2 Kaggle audit completion

**Evidence:** User supplied the v2 preflight and completed concise summary from Git commit `b2df70757ed57b53270913ecd453d5fc36a7ac12`. Saved the supplied fields in `experiments/runs/m0_answer_only_audit_20261105/reported_summary.json`, with provenance and preservation instructions in that folder's README. Preserved the user's executed notebook changes.

**Reported observations:** 1,260 completed predictions; zero parsed-answer matches (including zero year/non-year matches), zero format violations and zero token-limit hits. FP16 on Tesla T4; duration 38.4952 seconds; peak allocated GPU memory 1.325 GiB. No training occurred.

**Evidence limits:** The concise summary does not supply per-response answers, abstention/empty-answer counts or group details. Full files still reside at `/kaggle/working/m0-knowledge-audit-v2-n1fck5ni`. The reported prediction SHA-256 is `3bd2761acf1e45a4396fac5b9e5a0a86d01d01df1b1c5301d0d4ca7e08180f92`; it has not yet been independently verified.

**Files changed:** Added reported summary and run README; updated status, decision log, root README and evaluation-protocol state. No dataset, model or scoring changes.

**Next step:** User runs the notebook's archive cell and downloads the ZIP. Then preserve/verify all original v2 files and inspect a response sample before baseline/dataset acceptance. Full manual collision review remains pending.

---

## 2026-11-05 — Preserve and verify original v2 audit artifacts

**Work completed:** Located downloaded `m0-knowledge-audit-v2-n1fck5ni-results.zip`, checked ZIP integrity/exact five entries and saved all originals under `experiments/runs/m0_answer_only_audit_20261105/`. Added `artifact_verification.json`; preserved the supplied-summary transcription, original v1 audit and executed notebook.

**Verification:** Prediction SHA-256 matches `3bd2761acf1e45a4396fac5b9e5a0a86d01d01df1b1c5301d0d4ca7e08180f92`. Verified all 1,260 unique IDs/input fields against the candidate, all config/input/runner/parser hashes and the summary's embedded plan. Recomputed legacy and v2 score groups/counts, confirmed original review flags and the supplied concise summary. Logged termination fields are internally consistent; raw token IDs were not saved, so EOS cannot be independently replayed.

**Observed raw answers:** 1,259 exact `Answer: Unknown.` strings and one incorrect association for Cira Farrow (`ku_000017`): *The Last Days of the World* instead of *The Echoing Zenith*. All rows log EOS, generated-token counts span 5–18, and match/format/cap counts are zero. Inspected 42 first/middle samples across all 21 partition/task groups plus the sole non-Unknown exception.

**Interpretation:** Accept as the documented M0 behaviour under the v2 instruction. Most responses abstain; zero scores do not establish absent internal knowledge or forgetting. M1 must demonstrably learn the same targets before interpreting low M2 scores. No dataset changes or extra M0 runs are required merely to alter the baseline.

**Files changed:** Five original v2 output files, verification JSON, run README, root README, protocol state, status, decision log and this work log. No model, dataset or scoring-code changes.

**Next step:** Stop the Kaggle session, review/push the preserved records, finish candidate acceptance review and define/implement the tiny M1 learning pipeline with answer masking and adapter save/reload verification. Training remains unstarted.

---

## 2026-11-05 — Prepare bounded tiny M1 LoRA diagnostic

**Goal:** Prepare the first actual learning experiment on a fixed 12-unit subset, with answer-only masking, held-out wording, utility checks and adapter reload/persistence checks.

**Work completed:** Added training data utilities, a bounded training CLI, fixed diagnostic config, PEFT requirement, Kaggle/VS Code notebook, focused tests, pilot review/gate documentation and a preflight summary. Selected the first four F and first eight R units by sorted IDs, without model-score selection. Manually inspected the 12 canonical mappings and task/template definitions. External phrase searches produced unrelated results and were treated as inconclusive; admitted the subset for diagnostics only, not a full dataset release.

**Implementation:** Shared v2 chat wrapper/assistant targets; exact inference-token-prefix checks; prompt/padding label masking with supervised answer/EOS tokens; no length truncation; rank-8 all-linear LoRA on a frozen base; answer-normalized CE; AMP/optimizer update logging; before/after development and ten-question utility probes; changed-adapter and newly instantiated base reload comparisons. Notebook stages private HF upload and adapter-file hash round-trip before evidence export. No weights uploaded or runtime started in this local work.

**Evidence:** All 29 unit tests pass. CPU-only preparation with the actual pinned Qwen tokenizer passed all 84 examples: maximum length 101 tokens, 7,153 prompt tokens, 601 supervised answer/EOS tokens, exact prefix agreement and zero truncation. See `experiments/runs/m1_tiny_preflight_20261105/summary.json`. PEFT 0.21.2 package metadata was inspected for declared compatibility; CUDA import/training compatibility remains unverified.

**Files changed:** `src/training/`, `tests/test_training_data.py`, `configs/training/m1_tiny.json`, `requirements/m1-tiny.txt`, `experiments/notebooks/03_m1_tiny_learning.ipynb`, pilot documentation, preflight evidence and README/context state.

**Next step:** Push implementation, attach `HF_TOKEN` to a Kaggle GPU notebook and execute the diagnostic sequentially. Preserve failures as evidence. Passing it does not produce canonical M1 or validate RAFG/RASP; full-candidate release and expanded learning experiments remain future work.

---

## 2026-11-05 — Diagnose and correct first tiny M1 setup failure

**Evidence:** User confirmed Kaggle torch 2.11.0+cu130, Transformers 5.20.2 and PEFT 0.21.2 imports, private artifact access, and the same 84-row masking report. The training subprocess loaded Qwen, then failed at `get_peft_model` with `ImportError` for TorchAO 0.10.0 versus minimum 0.16.0. Code order and traceback place this before optimizer creation; zero optimizer updates occurred. Saved the reported failure and paths in `experiments/runs/m1_tiny_setup_failure_20261105/failure.json`.

**Cause:** Inspected PEFT v0.21.2 upstream import utilities and TorchAO LoRA dispatcher. Optional TorchAO presence triggers its version check before dispatch falls back to ordinary linear layers. It is unused by this unquantized pilot, so a full PyTorch/CUDA upgrade is unnecessary for this blocker.

**Correction:** Updated the notebook dependency cell to require a Kaggle runtime, uninstall only the unused `torchao` package there, and check optional backend absence in a fresh subprocess. Added a training-CLI compatibility check before base-model loading and documented the current-session recovery. Preserved all existing executed notebook outputs, the failed directory, canonical candidate and training settings.

**Verification:** Existing 29 tests pass; runner and edited notebook code compile; Git formatting checks pass. No local package removal or new model run performed. Runtime cleanup and actual LoRA training remain unverified until the Kaggle retry.

**Next step:** User applies the one-cell runtime cleanup/backend check, then reruns the training cell, which creates a new directory. Preserve and report its output; do not run upload cells for the incomplete attempt. Commit/push the durable fix with the next reviewed records.

---

## 2026-11-05 — Reported completed tiny M1 retry with failed pilot gates

**Evidence:** User supplied successful execution output after the TorchAO recovery, with progress for all 120 attempts and a completed summary at `/kaggle/working/m1-tiny-run-8myx_22r`. Saved its supplied result and rounded CE progress samples under `experiments/runs/m1_tiny_attempt_20261105/`. Full original artifacts and precise runtime state have not yet been imported.

**Reported observations:** F direct 8/8, F composite 11/12, R direct 16/16 and R composite 16/24. Base utility 0/10, post utility 4/10. Adapter changed, optimizer budget satisfied, zero format/cap issues and trained-versus-reloaded raw predictions identical. Pilot failed the preset R composite and measurable utility-baseline gates. Last printed training CE was 0.0007; no claim that loss alone establishes learning quality.

**Interpretation:** Learning and save/reload are demonstrated on this small, template-held-out diagnostic, while its full acceptance gates remain unmet. Base utility at zero cannot establish useful preservation; the cause requires raw-response inspection. Do not infer that a larger model, more steps or altered thresholds are needed from these aggregate results alone.

**Files changed:** Reported result JSON, CE-sample JSON and run README; root/status/decision/pilot-guide records updated. No model, dataset, training setting or threshold changed; all user notebook execution changes preserved.

**Next step:** Populate notebook `summary`, run private persistence/adapter-file hash verification, export the small evidence ZIP, and independently inspect raw metrics plus failed R composite and utility responses. Preserve the failed diagnostic and setup error separately; no canonical M1 is established.

---

## 2026-11-05 — Preserve and verify tiny M1 evidence; inspect failed outputs

**Work completed:** Located the downloaded `m1-tiny-run-8myx_22r-evidence.zip`, verified ZIP CRC and exact 14 entries, and preserved every original file without overwriting earlier transcriptions. Added a model-free `verify_artifacts.py`, `artifact_verification.json` and `manual_review.json` in `experiments/runs/m1_tiny_attempt_20261105/`. Updated the run README, root README, pilot guide, status and decisions; the executed notebook and training settings were not edited.

**Evidence:** Archive SHA-256 `239998bd1c0eec8685dc8e7fc0cf533c0ca6b09ccb400889ec1045f11b3fd548`. Candidate/input/config hashes and both recorded training-source hashes match commit `5ec5960`; current runner differs only by the already-recorded early TorchAO safeguard. All 282 prediction records rescore, 120 optimizer records and 12 printed loss samples agree, all summary metrics/gates reproduce, masking report matches prior preparation, and post/reloaded records are identical. Environment freeze confirms the pinned training stack and absent TorchAO. Saved private persistence receipt records adapter-file hash verification at revision `07d23d8b904ab48401f9c6676d05bb85a14d9fbd`. No local binary download or model inference replay occurred; token IDs were not exported, so EOS/count checks are internal consistency only.

**Diagnosis:** Base utility is ten exact `Answer: Unknown.` responses. Post utility has four strict matches, three correct numbers with units and three wrong answers; the separate manual semantic finding is 7/10 and does not replace 4/10. R composite failures comprise five author-to-year outputs (three correct years plus extra text) and three wrong prize-to-author outputs. All 11 development failures were inspected, including two one-hop failures outside the gate categories. Pilot remains failed; no claim of useful preservation, intrinsic arithmetic incapacity or proven prompt causation.

**Checks:** Evidence verifier passed; all 29 existing unit tests passed. No new GPU run, threshold/config/model change, commit or push.

**Next step / remaining work:** Define a separate utility-prompt comparison on the same pinned base and saved adapter before further SFT. This is proposed, not implemented/executed. Full-candidate release review, canonical M1/M2 and recovery remain pending.

---

## 2026-11-05 — Prepare inference-only utility prompt diagnostic

**Goal:** Diagnose the base model's ten arithmetic abstentions before another training attempt.

**Work completed:** Added `src/evaluation/utility_diagnostic.py`, fixed `configs/evaluation/utility_prompt_diagnostic_v1.json`, separate clean `04_m1_utility_prompt_diagnostic.ipynb`, protocol guide and two focused tests. Updated root README, tiny-pilot guide, status, decisions and this log. The completed training notebook, candidate, original scoring and preset pilot gates are unchanged.

**Design:** Restore the same private adapter at revision `07d23d8b904ab48401f9c6676d05bb85a14d9fbd`, verify pinned original evidence and adapter-file hashes, then compare the original-v2 versus arithmetic-specific system instructions on the same ten questions for base and adapter (40 predictions, zero updates). Save raw answers/rendered prompts/token IDs, strict scores, abstention/termination counts, provenance and exact original-output reproduction controls. A mismatch blocks interpreting instruction effects, rather than overwriting original evidence. The entire wrapper changes; this is not an isolated causal test of its Unknown clause or a trace evaluation.

**Local checks:** All 31 unit tests pass. New runner/test syntax and all six notebook code cells compile. Model-free plan passes with 40 planned predictions and zero updates; config reference hashes match preserved artifacts. No package install, network download, GPU evaluation, training, upload, commit or push performed.

**Next step / blocker:** New files must be committed/pushed before the notebook's GitHub clone can access them; this repository forbids pushing without authorization. Run the new notebook on Kaggle with HF_TOKEN attached, export/download the small evidence ZIP and inspect reproduction plus raw arithmetic answers. Actual remote download/loading/inference remains unverified; no new experimental results or canonical M1/M2 are claimed.

---

## 2026-11-05 — Authorize Git publication after diagnostic bootstrap failure

**Observed blocker:** The first utility notebook cell cloned commit `5ec5960` and stopped because the new diagnostic runner/config were still local. This was a source-availability assertion before model loading or inference, not a model failure.

**Authorization / scope:** User explicitly authorized committing and pushing the prepared diagnostic, preserved tiny M1/setup-failure evidence and context updates to `main`. Package the TorchAO safeguard, executed training/diagnostic notebook evidence, source/config/tests, small original run reports and protocol/context documentation; no adapters, optimizer binaries or credentials. Original dataset, model revision, scoring and failed pilot gates are unchanged.

**Checks:** All 31 unit tests and model-free evidence verification pass; both notebooks' code cells compile, Git whitespace checks pass, and prepared text files contain no common credential patterns or unexpected large binaries. No GPU diagnostic result is claimed. Git commit/push outcome is reported in the task response.

**Next step:** After successful publication, rerun notebook 04's first cell in the existing Kaggle runtime, confirm the new commit and continue sequentially. Preserve its evidence ZIP for verification.

---

## 2026-11-05 — Verify completed utility prompt diagnostic

**Work completed:** Confirmed all six notebook 04 code cells completed; source clone is `5d244def80cd70b0407df143322d092b1a2b48e7`, published after the user's prior authorization. Located `m1-utility-prompt-9ypiagyk-evidence.zip` in Downloads, verified CRC/exact eight entries and preserved original files without overwrite under `experiments/runs/m1_utility_prompt_diagnostic_20261105/`. Added model-free verification script, artifact report, paired manual review and run README. Updated root README, utility/tiny-pilot guides, status, decisions and this log; executed notebook outputs remain preserved.

**Evidence:** Archive SHA-256 `40f8efa30e59f90ed63382f2ece882b47a5cbc5215dc7f69dbdbd669d17b84df`. All forty input/score records, prediction/config/source hashes, wrapper/question prompt strings, group metrics/abstentions, environment versions and both original-v2 exact-response controls verify locally. Remote source-adapter hash receipt agrees with the preserved original adapter manifest. Saved generated token counts/end markers are consistent; pinned model EOS identity/tokenizer rendering was not independently replayed. No local binary re-download/model execution occurred.

**Results:** Original-v2 base 0/10 (ten Unknown), adapter 4/10; arithmetic-v1 base 7/10 (zero Unknown), adapter 6/10 (zero Unknown). Both checkpoints under arithmetic-v1 make the same three numerical errors. Adapter's extra strict mismatch is correct `12 kilometers` versus bare `12`; manual numerical correctness is 7/10 for both. All groups have zero format/cap violations. Reported inference-section duration 4.2309 seconds on Tesla T4; zero training updates or traces evaluated.

**Interpretation / remaining work:** Full-wrapper sensitivity is demonstrated on ten probes, not the effect of the Unknown clause alone. No general utility-preservation/degradation claim; arithmetic-v1 base remains below the original >=8/10 gate. Original pilot remains failed, with R composite failures unresolved; no canonical M1/M2. Define the next utility protocol/probe prospectively before another training run. No model, threshold, scoring or training-setting changes, new commit or push performed during this review.

**Checks:** The evidence verifier passed with 40 rescored records and exact original-output reproduction; original archive/file preservation checked. New verifier syntax and Git whitespace checks pass. No production code changed or additional GPU run initiated.

---

## 2026-11-05 — Prepare matched-sampling thinking-mode diagnostic

**Goal:** Test whether the existing model's thinking mode helps arithmetic and relational-development errors before another SFT run or model change.

**Work completed:** Added `src/evaluation/mode_diagnostic.py`, fixed `configs/evaluation/thinking_mode_diagnostic_v1.json`, clean `05_m1_thinking_mode_diagnostic.ipynb`, protocol guide and six focused tests. Reused existing reference evaluators, scoring, utility examples, adapter provenance and development metrics. Updated README, status, decisions and this log; original notebooks, dataset, training config, scores and gates remain preserved.

**Design:** Greedy reproduction controls plus sampled non-thinking/thinking at matched temperature 0.6, top-p 0.95, top-k 20, min-p 0 and 1,024-token caps. This follows the pinned Qwen model card's warning against greedy thinking. Ten utility probes on base and adapter plus all 84 development questions on adapter yield 312 planned predictions; zero updates and no final-test/N examples. Save visible reasoning and final answers separately, preserve thinking delimiters even if special tokens, reject incomplete/malformed thinking before answer acceptance, and persist sampled results after each batch. Record seed/template/termination/token metadata and greedy reproduction controls. One seed and capped generations limit interpretation; a thinking-cap hit does not establish absent reasoning ability.

**Local checks:** All 37 tests pass, including trace-only target rejection, complete/prefilled boundaries, missing/malformed boundaries, unchanged strict non-thinking scoring, special-token delimiter preservation and exact full development selection/reference hashes. Source/new notebook syntax and model-free plan pass (312 total/104 thinking predictions, zero updates). No model weights, remote runtime, GPU inference or training executed locally.

**Next step / blockers:** New code/config and previously verified utility reference evidence must be published to GitHub before notebook 05's clone can use them; no new push authorization requested or exercised during implementation. After publication, use existing Kaggle GPU/HF_TOKEN, run sequentially and preserve the small evidence ZIP. No actual thinking-mode improvement, utility pass, canonical M1/M2 or recovery is claimed.

---

## 2026-11-05 — Authorize publication of thinking-mode diagnostic and utility evidence

**Authorization / scope:** User explicitly approved committing and pushing notebook 05, supporting source/config/tests/protocol, verified notebook 04 evidence and context updates to `main`. Preserve the executed notebook 04 outputs. No model weights, credentials, dataset/scoring changes or GPU run included.

**Checks:** All 37 tests and the 40-record utility evidence verifier pass; pending notebook code compiles, Git whitespace checks pass, and prepared text files are checked for common credential patterns and unexpected large binaries. Thinking-mode runtime remains unverified. Commit/push outcome is reported in the task response.

**Next step:** Run notebook 05 from the first cell on the existing Kaggle GPU kernel after successful publication, confirm the printed new source commit, run sequentially and download its evidence ZIP. No training or unlearning is required for this diagnostic.

---

## 2026-11-05 — Verify completed thinking-mode diagnostic and inspect traces

**Work completed:** Confirmed all six notebook 05 code cells ran from commit `3c77324a4c70f987074786ec3810230a3fb53bac`. Verified `m1-thinking-mode-ua9h7f7s-evidence.zip` CRC/exact fourteen entries, preserved all original reports without overwrite in `experiments/runs/m1_thinking_mode_diagnostic_20261105/`, and added an independent verifier, artifact report, full paired-final-answer/manual trace review and run README. Updated root README, tiny/thinking guides, status, decisions and this log. Executed notebook outputs preserved.

**Evidence:** Archive SHA-256 `e638d0b8f3a3775953375074ba32087ddf53edadd2218ef80e1489e37b3359c6`. All 312 input/score records, nine prediction hashes, config/source hashes, sampled prompt/seed conditions and summary metrics verify. Three greedy references reproduce exactly. 228 saved token records agree with declared EOS/think metadata; 84 original greedy development rows have no token IDs. Environment versions and private adapter-hash receipt agree; no local inference/tokenizer replay/binary download occurred. Reported duration 271.4286 seconds, Tesla T4, zero updates.

**Observed results:** Matched arithmetic base 7/10 non-thinking to 10/10 thinking; adapter 6/10 to 10/10. Three base thinking finals omit Answer: but values match, logged as format violations; all twenty thinking utility answers complete without caps. Adapter development changes 73/84 to 21/84, with 53 paired losses/one gain/twenty correct both/ten wrong both. Thinking direct F=2/8, R=3/16; composite F=3/12, R=6/24. One capped/unclosed development response repeats Meteor; 62 other thinking failures complete. Non-thinking sampled R composite is 18/24, without replacing the original greedy 16/24 gate.

**Manual diagnosis:** Inspected all 84 thinking development final answers, all thinking utility finals, five selected failed fact traces, six arithmetic traces for previously wrong problems and the capped trace. Observed invented/mixed associations, two correct years with extra text, and a trace/final contradiction. Thinking helps arithmetic here but weakens learned fictional-fact output. SFT/evaluation mode mismatch is plausible, not causally isolated. This inference run is not unlearning or evidence of erased facts.

**Checks / next step:** Independent evidence verifier passes all 312 rows/228 token records and three exact controls; new verifier syntax and Git whitespace checks pass. No production code, training settings, model size, thresholds, commit or push changed during review. Original tiny pilot remains failed; canonical M1/M2 and recovery are pending. Design a versioned tiny retry with intended training/evaluation modes fixed prospectively before scaling or another experiment.

---

## 2026-11-05 — Prepare fresh-base mixed-mode tiny M1 training retry

**Goal:** Test learning fictional facts with both answer-only and explicit thinking representations after notebook 05 exposed mode-sensitive fact access.

**Work completed:** Added canonical-chain target renderer `src/training/mixed_data.py`, bounded runner `src/training/mixed_m1.py`, fixed config `configs/training/m1_tiny_mixed_v1.json`, clean notebook `06_m1_tiny_mixed_learning.ipynb`, protocol guide, five focused tests and local preparation evidence. Updated README/status/decisions/log. Reused existing fixed subset selector, masking/padding, original greedy evaluator/gates, sampled thinking evaluator, scoring and utility questions. Original candidate, training implementation/config, adapters and failed gates remain preserved.

**Design:** Fresh pinned base/new rank-8 all-linear LoRA; same original seed/batch/learning rate/120 attempts. Double each of 84 train questions into non-thinking answer-only and thinking canonical-chain targets (168 rows); no utility/dev/final question training. Synthetic chains use only the corresponding canonical training fact path and are not model-generated thoughts. Supervise full response/EOS, including rationale. Evaluate 84 development questions in each mode and ten thinking utility questions at base/post/reload (534 predictions if baseline passes). Require exact saved base-utility thinking reference reproduction and measurable baseline before adapter creation. Save adapter/optimizer/manifest before long post evaluation; preserve privately and verify hashes even if later gates fail.

**Local evidence:** All 42 tests pass. Actual cached pinned Qwen tokenizer preparation via `/private/tmp/pllu-data-audit-venv` verifies all 168 prefixes with no truncation: answer-only max101, prompt7153/response601 tokens; thinking max134, prompt6817/response2298 tokens. Evidence in `experiments/runs/m1_tiny_mixed_preflight_20261105/` includes plan, selected targets/dev, masking and summary. No model weights/PyTorch/GPU loaded, no training/inference/remote upload performed. Runner/new notebook syntax, model-free plan and Git whitespace checks pass.

**Limits:** No learning result yet. Same update attempts with doubled rows/longer supervision are not token/compute matched to original, so this is a combined representation retry, not an isolated causal ablation. Multi-hop answers/chains are supervised; held-out wording does not prove unseen composition. Ten arithmetic questions are now repeatedly inspected development probes. Original pilot remains failed; no canonical M1/M2 or final benchmark protocol selected.

**Next step / blocker:** Publish new source/config/notebook plus verified notebook 05 reference evidence with explicit user authorization, then execute notebook 06 on Kaggle with HF_TOKEN. Back up adapter/checkpoint metadata privately, verify round-trip hashes, download evidence ZIP and independently verify both-mode learning, utility, update exposures and reload before any scaling. No commit/push performed during preparation.

---

## 2026-11-05 — Authorize publication of mixed-mode tiny training retry

**Authorization / scope:** User explicitly approved committing/pushing notebook 06, its source/config/tests/protocol/local tokenizer preflight, verified notebook 05 evidence and context updates to main. Preserve executed notebook 05 outputs. No model/optimizer binaries, credentials or GPU run included; original candidate/answer-only implementation/config/gates remain unchanged.

**Checks:** All 42 tests pass; thinking-mode evidence verifier rescored all 312 records and checked 228 token records/three greedy controls. New notebook code compiles, prepared files are checked for common credential patterns/large binaries, and Git whitespace checks pass. Actual mixed-mode training is pending. Commit/push outcome is reported in the task response.

**Next step:** Open notebook 06 on the existing Kaggle GPU kernel after publication, confirm the new source commit and run sequentially. This notebook performs actual fresh-adapter training after the base utility/reference gate, then private persistence/hash verification and small evidence export. Preserve failed results too; no canonical M1/M2 is claimed.

---

## 2026-11-05 — Verify completed mixed-mode tiny M1 retry

**Work completed:** Confirmed all eight notebook 06 code cells completed from `7d4b28f1d18d543ea0daacf8f09cb2df9dbf0eed`. Located `m1-tiny-mixed-v1-_avmkde0-evidence.zip`, verified CRC/exact twenty entries and preserved original bytes without overwrite under `experiments/runs/m1_tiny_mixed_attempt_20261105/`. Added independent verifier, archive verification, detailed failure/trace review and run README. Updated README/mixed protocol/status/decisions/log; executed notebook outputs preserved.

**Evidence/checks:** Archive SHA-256 `4eabb6f0b6374cafddbf29ac37cb3b0fa53f250d9af0db8e71216de8d94b1632`. All source/config/reference hashes, 168 constructed training rows/84 dev inputs, masking preflight, 534 scores/nine group reports, 282 token records and full candidate verdict reproduce. Baseline utility raw strings reproduce the old thinking reference. Post/reload rows match except batch_seconds, including token IDs. All 120 updates and per-mode CE/count arithmetic verify; logged examples 962 answer-only/958 thinking (1920 total). Detailed exposure token totals are runtime reports, not independent data-loader replay. Saved private receipt verifies adapter hashes at revision `25764276e3e83e6c20519429854e15216f57e676`, path `diagnostics/m1-tiny-mixed-v1/m1-tiny-mixed-v1-_avmkde0`. No local binary download/inference replay.

**Results:** Base facts 0/84 in each mode; post answer-only 78/84, thinking 63/84; thinking utility 10/10 before/after. Answer-only gate set passes. Thinking F direct 7/8 and R composite 16/24 fail; thinking R direct 15/16 and F composite 10/12 pass. One F prize-to-year loops to cap without closing/final answer; corresponding cap/boundary/format gates fail. Overall mixed pilot remains failed, not canonical M1/M2. Reported training 229.4284 seconds, full run 607.7479 seconds, Tesla T4, peak 3.3516 GiB; no speed-comparison claim.

**Manual diagnosis:** Inspected all six answer-only failures, twenty completed failed thinking traces/finals, capped tail and all post arithmetic finals. Thinking failures: six author-to-year, eight prize-to-year, four prize-to-author, two book-to-author, one book-to-prize. One correct year includes extra prize text; other examples include wrong years/authors/type and invalid prize-to-prize chains. Response CE 0.0023 at last update does not prove generation generalization. The descriptive 21/84 old-adapter versus 63/84 mixed-adapter difference is not an isolated causal ablation.

**Next step/limits:** Define inference-only training-wording versus development-wording comparison on this same saved adapter with prespecified modes/seeds before changing SFT. No such diagnostic is implemented/run yet. No gate lowering, selective record exclusion, additional updates, new training setting, commit/push or unlearning performed during review. Model-free evidence verifier and Git whitespace checks pass; no production source changed.

---

## 2026-11-05 — Prepare same-adapter training/development wording comparison

**Goal:** Distinguish training-question generation from held-out-wording failures before another SFT change.

**Work completed:** Added clean `07_m1_training_wording_diagnostic.ipynb`, review config `configs/evaluation/training_wording_diagnostic_v1.json` and `docs/data/TRAINING_WORDING_DIAGNOSTIC.md`. Updated root/run README, status, decisions and this log. Reused existing dataset selectors, canonical target validation, greedy/sampled evaluators, scorer and metadata helpers without production source changes.

**Design:** Same private mixed adapter/revision, all 84 train/development record-task pairs, same gold answers and aligned order; two wordings in each of two declared modes plus ten thinking utility controls (346 predictions, zero training). Hash-verify original reports/adapter and reproduce all original development/utility raw responses before training-wording inference. Save paired categories/all final responses, raw traces/tokens/flags, source/config/version/adapter provenance and evidence ZIP. Release model GPU memory after evaluation. One sampled seed; no final tests, new gates or pilot promotion.

**Execution choice:** Local VS Code notebook embeds the versioned config and orchestrates already-published helpers pinned at `7d4b28f`; immutable private HF supplies the previously verified mixed-run evidence. New notebook/config publication is therefore not a prerequisite for remote execution. Only the project's src import cache is reset when selecting the pinned temporary clone, avoiding stale modules from earlier notebooks. No Git commit/push or new model upload is performed automatically.

**Local checks:** All seven code cells compile. Executed the model-free plan cell locally in a temporary directory: 84 exact pairs, distinct questions/exclusive templates, matching embedded/config-file values, six preserved reference hashes and helper hashes agreeing with published commit. All 346 predictions remain planned, not generated. No new test suite or GPU/model run was executed; existing production implementations are unchanged. Git whitespace and common-credential/large-file checks pass.

**Next step:** User can run notebook 07 now on the existing Kaggle GPU kernel/HF_TOKEN, without waiting for another source push, and download the evidence ZIP. If training wording is strong while development is weak, investigate wording/generalization; if both fail, investigate generation/learning reliability. These are hypotheses, not observed outcomes. Original mixed-mode pilot remains failed; canonical M1/M2 pending.

---

## 2026-11-06 — Verify completed training/development wording diagnostic

**Work completed:** Confirmed all seven notebook 07 code cells completed using pinned published `7d4b28f` helpers and saved mixed adapter `25764276e3e83e6c20519429854e15216f57e676`. Located `m1-wording-probe-hlq0680r-evidence.zip`, checked CRC/exact fourteen entries, preserved original bytes without overwrite in `experiments/runs/m1_training_wording_diagnostic_20261106/`, and added independent verifier, archive verification, failure/paired interpretation and run README. Updated root/run README, wording protocol, status, decisions/log; executed notebook preserved.

**Evidence/checks:** Archive SHA-256 `9048b324c3a18fcf8167304e32bb3983b519943bcb9a70b43452b394ff11225c`. All 346 input/score records, 178 saved token records, five summary groups, three exact original raw controls and all 168 paired response/category records reproduce. Config/selector/published helper hashes, private reference/adapter-manifest receipt, exact 84 pairs/exclusive templates and environment versions agree. No local binary download/model/tokenizer replay. Reported evaluation-section time 139.9381 seconds, Tesla T4; UTC November 5 corresponds to November 6 IST. No speed claim.

**Observed results:** Exact training wording 80/84 for answer-only and thinking; development remains 78/84 and 63/84; utility 10/10. Thinking pairs: 63 correct both, seventeen training-only, zero development-only, four wrong both. Answer-only pairs: 77/3/1/3. Training groups have no caps/format/boundary failures. Development thinking reproduces its one existing capped/unclosed response. All four training failures in each mode and all discordant pair identities/questions were inspected; residual errors concern years/reverse lookup, not just formatting.

**Interpretation/next step:** Supports wording-sensitive generation, particularly thinking, without establishing perfect memorization or sole causal mechanism. Keep mixed-mode pilot failed; canonical M1/M2 pending. Next proposed concrete correction is a separately versioned fresh-base tiny mixed SFT retry with new training-only paraphrases and a declared exposure budget, preserving candidate, held-out prompts, old adapters and thresholds. No augmentation implementation, new hyperparameters, training, commit or push performed in review. Evidence verifier and Git whitespace checks pass; production source unchanged.

---

## 2026-11-06 — Prepare fresh-base tiny training-paraphrase retry

**Goal:** Make the concrete training change supported by notebook 07's observed wording gap, while preserving all evaluation data/gates and prior baselines.

**Work completed:** Added fourteen newly authored train-only templates, `src/training/paraphrase_data.py`, fixed `configs/training/m1_tiny_paraphrase_v1.json`, clean `08_m1_tiny_paraphrase_learning.ipynb`, protocol, five tests and cached-tokenizer preflight evidence. Extended the existing mixed runner only through an explicit new run type/data overlay and optional post/reload original-training recall probes; original run type/targets/534-prediction plan remain unchanged. Updated README/status/decisions/log. No canonical data/generator/evaluation/scorer/old checkpoint changes.

**Design:** Three wordings per original question (original +two new), both original response modes, same twelve records/seven tasks/target chains =252 questions/504 rows. New bank rejects non-source placeholders, original/reserved template reuse/near-duplicates, answer leakage, duplicate questions/IDs and development equality. Maximum observed reserved-template similarity 0.5679012346; original bank consulted for exclusion only. Same pinned fresh base/new rank8 LoRA, seed42/batch4/accum4/LR0.0003/120 attempts and all gates. Base178/post346/reload346 =870 planned predictions; training recall probes are diagnostic only. No automatic budget extensions, evaluated dev-question copying, final-test tuning or whole-pilot promotion from a passing subset.

**Local checks:** All 47 tests pass, including unchanged old plan/targets/dev and optimizer/gate constraints. Actual cached pinned tokenizer prepares all504 prefixes/masks with zero truncation: answer-only max106, prompt22503/response1803; thinking max138, prompt21495/response6894. Source/new notebook syntax and model-free plan pass. Existing mixed and wording evidence verifiers still pass (534/346 records). Preflight evidence is in `experiments/runs/m1_tiny_paraphrase_preflight_20261106/`. No model weights, PyTorch/GPU training/inference or remote upload performed locally.

**Limits/next step:** Same attempts over more forms do not imply equal per-surface or token/compute exposure; prior 80/84 training recall is not perfect memorization. New adapter learning is untested. Publish prepared files and verified pending evidence with user authorization, run notebook08 on Kaggle/HF_TOKEN, preserve private adapter/round-trip receipt and ZIP, then independently verify all gates and recall outputs. Original/mixed pilots remain failed; canonical M1/M2 pending. No commit/push performed during implementation.

---

## 2026-11-06 — Authorize publication of training-paraphrase retry

**Authorization/scope:** User explicitly approved committing/pushing notebook08, source/config/template bank/tests/protocol/preflight, verified pending mixed-run and wording-run evidence, notebook06/07 execution records and context updates to main. Preserve earlier adapters/data/gates; no model/optimizer binaries or credentials included.

**Checks:** All 47 unit tests pass. Mixed-run and wording-run verifiers reproduce all 534/346 records and their stored controls/gates. Pending notebook code compiles; common-credential/large-file and Git whitespace checks pass. Augmented GPU training remains pending. Commit/push outcome is reported in the task response.

**Next step:** Open notebook08 in the existing Kaggle GPU kernel/HF_TOKEN, confirm the new source commit and run sequentially. It trains a fresh adapter after baseline checks, then verifies private backup and exports small evidence. Preserve failed outcomes; no canonical M1/M2 claimed.

---

## 2026-11-06 — Verify completed tiny training-paraphrase retry

**Work completed:** Confirmed all eight notebook08 code cells completed from `c6f53fe0475b220c8eab1a29dd93336db3507661`. Verified `m1-tiny-paraphrase-v1-0ar28u5m-evidence.zip` CRC/exact25 entries and preserved original bytes under `experiments/runs/m1_tiny_paraphrase_attempt_20261106/`. Added adapted independent verifier, archive verification, all-post-failure/recall/utility review and README. Updated root README, paraphrase protocol, status, decisions/log; executed notebook preserved.

**Evidence/checks:** Archive SHA-256 `de31fb8f428ffb9f2f02c9a0386425e6e9bc193c0bbaf1ae9b787ad6878a8a8e`. All13 prediction hashes, source/config/bank hashes,504 canonical targets/audit, unchanged dev data, tokenizer preparation,870 scores/stage groups and450 token records verify. Base utility raw strings reproduce original reference. All120 updates, CE/count arithmetic and1920 examples agree (953 answer-only/967 thinking); detailed token totals are runtime reports, not data-loader replay. Post/reload rows match except batch_seconds, including tokens. Private adapter receipt records verified hashes at revision `f8fbea97a97cc50f3572c3efb62219dbf5820fa9`, path `diagnostics/m1-tiny-paraphrase-v1/m1-tiny-paraphrase-v1-0ar28u5m`. No local model/tokenizer replay/binary download.

**Results:** Development77/84 answer-only,70/84 thinking; original-training probes77/84 and76/84. Utility base10/10→post9/10 is exactly the allowed drop. All direct author/book queries, R composite and post format/cap/boundary checks pass. Thinking F composite6/12 (<required9/12) is the only false gate. All six failed F composites are completed wrong answers (three prize-to-author/three author-to-year); no threshold changes. Overall pilot remains failed, not canonical M1/M2.

**Tradeoffs/manual review:** Inspected all post dev failures, training-probe failures and sole arithmetic error (chair calculation5 instead of15). Compared with mixed v1, aggregate thinking improves63→70, but F composite10→6, original-training recall80→76 thinking, and utility10→9. Narrower wording gap partly comes from lower training recall. More question forms at the same120 attempts mean fewer repeats per surface; exposure adequacy is a hypothesis, not a proven cause. Reported training228.2085 seconds/full488.3965 seconds, Tesla T4, peak3.4322 GiB; no speed claim.

**Next step/limits:** Propose a separately versioned bounded longer-budget comparison using unchanged504 rows/model/LoRA/evaluation/gates and preserving120-step control. No new budget/config/longer run, model change, gate lowering, selective exclusion, unlearning, commit or push performed during review. Evidence verifier and Git whitespace checks pass; production source unchanged.

---

## 2026-11-06 — Prepare pinned Qwen3-1.7B same-recipe tiny comparison

**Goal:** Test candidate model suitability rather than assuming 0.6B is sufficient or that size explains every failure.

**Work completed:** Added `src/training/model_capacity.py`, fixed `configs/training/m1_tiny_qwen17_v1.json`, clean notebook09, protocol, five focused tests and public metadata/tokenizer preflight evidence. Extended the mixed runner only with a new explicit model-comparison run type, disposable fit mode, own-model baseline policy and comparison report; old protocols/data/gates unchanged. Updated project/model context, README/status/decisions/log. 0.6B configs/adapters remain controls.

**Model identity/data:** Public HF API verifies `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. Retrieved only public metadata/config/tokenizer, using temporary cache with no safetensor model weights; first sandboxed network attempt failed DNS, approved escalation succeeded. Metadata under `experiments/runs/qwen17_metadata_20261106/`; exact 504 training targets/84 dev questions/recipe/gates match 0.6B. The new baseline measures its own M0 outputs and cannot read 0.6B response expectations as its utility baseline. Full new-model candidate audit remains future if selected.

**GPU plan:** Separate fit process exercises longest actual training batch/AMP/one temporary AdamW update and representative batch16 thinking generation; saves memory metadata, no adapter/checkpoint or learning metrics. Real trial then starts fresh from pinned base/new LoRA with 120 attempts, own baseline guard and 870 evaluations including recall/reload. Preserve 0.6B control and report measured tradeoffs; no compute-matched or size-only causal claim. No automatic fallback or larger-model success assumption.

**Local checks:** All 52 tests pass, including strict model pin, unchanged recipe/data/gates and independent new baseline acceptance/rejection. Actual cached 1.7B tokenizer verifies all504 prefixes/masks/no truncation, max106/138 and prompt22503/21495, response1803/6894 tokens. All nine notebook cells/source syntax and model-free plan pass; existing 870-row 0.6B verifier still passes. No model weights/PyTorch/GPU fit/inference/training loaded/run locally. Preflight under `experiments/runs/m1_tiny_qwen17_preflight_20261106/`.

**Next step/limits:** Publish new source/config/public metadata/verified 0.6B control evidence with authorization, run notebook09 on Kaggle/HF_TOKEN and preserve both fit and real-trial evidence/private receipt. Representative fit is not a guarantee against worst-case OOM. 0.6B longer-budget proposal is deferred; final model choice, full release, canonical M1/M2 and recovery remain unestablished. No commit/push or actual 1.7B result during implementation.

---

## 2026-11-06 — Authorize publication of Qwen3-1.7B comparison

**Authorization/scope:** User explicitly approved committing/pushing notebook09, its source/config/tests/protocol/public metadata/tokenizer preflight, verified notebook08 control evidence/executed notebook and context updates to main. No model/optimizer binaries, credentials or 1.7B GPU run included; all earlier model configs/adapters/gates remain preserved.

**Checks:** All 52 tests pass. The 0.6B augmentation verifier reproduces all 870 scores/450 token records/120 update records and the failed verdict. Pending notebook code compiles; common-credential/large-file and Git whitespace checks pass. GPU fit, new-model M0 inference and real 1.7B training remain untested. Commit/push outcome is reported in the task response.

**Next step:** Open notebook09 on the existing Kaggle GPU kernel/HF_TOKEN after publication, confirm the new source commit, then run sequentially. Disposable fit metadata is saved separately; its temporary update/checkpoint state is not reused by actual training. Preserve fit/private adapter/evidence even on failure. Final model selection/canonical M1/M2 remain pending.

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
