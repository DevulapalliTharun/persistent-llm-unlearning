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
