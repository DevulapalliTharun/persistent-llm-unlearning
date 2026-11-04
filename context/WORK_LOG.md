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
