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
