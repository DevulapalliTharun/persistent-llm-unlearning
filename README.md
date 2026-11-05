# Persistent LLM Unlearning — RAFG & RASP

M.Tech major-project research workspace for studying whether **targeted knowledge unlearning stays effective during later legitimate reasoning fine-tuning**.

**Current state:** The 0.6B tiny pilot remains failed at thinking F composite. A same-recipe Qwen3-1.7B comparison is prepared with a disposable GPU memory probe and its own M0 baseline, preserving the 0.6B control. All 52 local tests and actual pinned-tokenizer preparation pass; GPU fit and learning are untested. Final model choice and canonical M1/M2 remain pending.

## The two candidate approaches

| Method | Strategy | Basic idea |
|---|---|---|
| **RAFG** — Reasoning-Aware Forgetting Guard | **Reactive** | Monitor whether designated forgotten answers become easier to predict; trigger short corrective forgetting updates only when necessary. |
| **RASP** — Retain-Aware Selective Plasticity | **Preventive** | Estimate which LoRA modules are sensitive to forget targets relative to retain targets; constrain those modules while allowing others to learn. |

Both test **post-unlearning safeguards**. Neither is proposed as a novel base-model architecture or a new original machine unlearning algorithm.

## Shared workflow

```text
M0 (base model)
 └── Supervised fine-tuning on invented facts
       ↓
M1 (learned fictional facts)
 └── Existing Gradient Difference unlearning
       ↓
M2 (canonical post-unlearning checkpoint)
 ├── No-protection fine-tuning: demonstrate/measure recovery
 ├── RAFG fine-tuning: monitor and correct when triggered
 └── RASP fine-tuning: score modules and selectively regularize
```

**Important:** Save one validated M2 checkpoint and start every continuation branch from an identical independent copy. Demonstrate an ordinary recovery challenge before claiming either safeguard is effective.

## Repository layout

```text
persistent-llm-unlearning/
├── context/
│   ├── PROJECT.md      # Research problem, methods, data and metrics
│   ├── DECISIONS.md    # Confirmed choices versus proposals
│   ├── STATUS.md       # Current stage, open gates and next steps
│   └── WORK_LOG.md     # Dated planning/experiment evidence
├── docs/
│   ├── data/
│   │   └── DATASET_SPEC.md
│   └── proposals/
│       ├── RAFG.pdf
│       └── RASP_Project_Guide.pdf
├── configs/            # Versioned dataset configuration
├── data/               # Full pilot candidate, manifests and checksums
├── src/                # Data generation, validation and M0 smoke-test code
├── tests/              # Automated generator and audit tests
├── experiments/        # Notebooks and recorded validation/run evidence
├── references/         # Notes and related-work references (future)
├── AGENTS.md           # Instructions for coding assistants
├── .gitignore
└── README.md
```

## Development, execution and storage architecture

The Mac and future Dell are development clients, not training machines. Source code and project memory move through GitHub; GPU work runs on Kaggle, with Colab as a fallback; durable model artifacts live in the private Hugging Face model repository `tharundevulapalli/persistent-llm-unlearning-checkpoints`.

```mermaid
flowchart TB
    subgraph Clients["Development clients"]
        MAC["Company Mac<br/>VS Code + Git client"]
        DELL["Personal Dell<br/>VS Code + Git client"]
    end

    subgraph Control["Versioned project control plane"]
        GH["GitHub repository<br/>code · configs · context · small metrics"]
    end

    subgraph Compute["Remote execution — ephemeral"]
        KG["Kaggle Jupyter GPU runtime<br/>primary training environment"]
        COLAB["Google Colab GPU runtime<br/>fallback environment"]
        KSEC["Kaggle / Colab Secrets<br/>HF_TOKEN and other runtime credentials"]
        TEMP["Runtime working storage<br/>cloned repo · caches · temporary checkpoints"]
    end

    subgraph Storage["Persistent ML artifact storage"]
        HFBASE["Hugging Face public model hub<br/>base model + tokenizer"]
        HFPRIV["Private Hugging Face model repo<br/>tharundevulapalli/persistent-llm-unlearning-checkpoints<br/>M1/M2 adapters · resumable checkpoints · metadata"]
        KOUT["Saved Kaggle notebook versions<br/>logs and recoverable run outputs"]
    end

    MAC <-->|"git pull / reviewed commits + push"| GH
    DELL <-->|"git pull / reviewed commits + push"| GH
    MAC -.->|"VS Code notebook<br/>remote-kernel connection"| KG
    DELL -.->|"VS Code notebook<br/>remote-kernel connection"| KG
    GH -->|"clone / pull exact commit"| KG
    GH -->|"clone / pull exact commit"| COLAB
    KSEC -->|"inject at runtime; never write to Git"| KG
    KSEC -->|"inject at runtime; never write to Git"| COLAB
    HFBASE -->|"download pinned M0 revision"| KG
    HFBASE -->|"download pinned M0 revision"| COLAB
    KG <-->|"save / restore private artifacts"| HFPRIV
    COLAB <-->|"save / restore private artifacts"| HFPRIV
    KG --> TEMP
    COLAB --> TEMP
    TEMP -->|"explicit Save Version when useful"| KOUT
    KOUT -->|"restore or download"| KG
    KG -->|"download small reviewed results"| MAC
    COLAB -->|"download small reviewed results"| MAC
    MAC -.->|"later handoff through GitHub + artifact IDs"| DELL

    classDef client fill:#fff1ad,stroke:#c5a642,color:#263238,stroke-width:1.5px;
    classDef control fill:#d9eaff,stroke:#4c8bc7,color:#263238,stroke-width:1.5px;
    classDef compute fill:#dcecff,stroke:#4c8bc7,color:#263238,stroke-width:1.5px;
    classDef secret fill:#ffe0bf,stroke:#d58a32,color:#263238,stroke-width:1.5px;
    classDef system fill:#d8f0d5,stroke:#4b9b58,color:#263238,stroke-width:1.5px;
    class MAC,DELL client;
    class GH control;
    class KG,COLAB,TEMP compute;
    class KSEC secret;
    class HFBASE,HFPRIV,KOUT system;
```

### What is stored where

| Location | Store here | Do not store here | Access direction |
|---|---|---|---|
| **GitHub repository** | `src/`, experiment configs, notebooks, deterministic data-generation code, context files, small manifests, metrics and plots | Secrets, access tokens, base-model weights, large checkpoints or caches | Mac/Dell push reviewed changes; Kaggle/Colab pull an exact commit |
| **Kaggle or Colab runtime** | Active clone, downloaded model cache, generated working data and temporary checkpoints | The only copy of any important result | Runtime reads GitHub/Hugging Face; important outputs must leave the runtime before shutdown |
| **Private Hugging Face model repository** | M1 adapter, canonical M2 adapter, resumable adapter checkpoints, tokenizer additions if any, run metadata and checksums | Source-of-truth project documentation or unreviewed secrets | Kaggle/Colab upload after milestones and download to resume; Mac/Dell may verify or download |
| **Saved Kaggle notebook output** | Versioned logs and run outputs useful for recovery or audit | Sole canonical M1/M2 copy | Kaggle creates a saved version; later sessions can attach or download it |
| **Mac / Dell** | Working Git clone and optional downloaded backup | Sole copy of project memory or checkpoints | Synchronize through GitHub and the private artifact repository |
| **Kaggle / Colab Secrets** | Narrowly scoped `HF_TOKEN` and any required private-repository credential | Tokens in notebook cells, scripts, logs or Git | Credentials are injected into the runtime only when needed |

### Run lifecycle and access rules

1. In VS Code, pull `main`, read `AGENTS.md` and `context/`, then make and review source/config changes.
2. Commit and push the reproducible code state to GitHub. A run must record the exact Git commit SHA.
3. Start a Kaggle GPU session and connect the local VS Code notebook to its remote Jupyter kernel, or use Colab as the fallback.
4. In the runtime, clone/pull the recorded Git commit and download the pinned base-model revision from Hugging Face.
5. Read private credentials only from the platform secret store. Use a narrowly scoped token and never print it.
6. Run the experiment. Treat runtime disk as disposable even when a platform offers saved outputs.
7. At a verified milestone, upload the adapter/checkpoint and metadata to the private Hugging Face repository. M2 is not considered saved until it can be restored in a fresh session.
8. Bring small metrics, plots and logs back for review; update `STATUS.md`, `WORK_LOG.md` and, when needed, `DECISIONS.md`; then commit and push those records.
9. During the Dell handoff, clone GitHub and restore artifacts by their recorded Hugging Face repository path, revision and checksum. No dependency on the company Mac should remain.

The runtime may access the public base model without a write credential. Access to the private artifact repository requires a private token stored in the runtime's secret manager. Kaggle/Colab should not push project changes directly to GitHub in the normal workflow; changes return through the reviewed local working copy.

**Verified storage check (2026-11-04):** A committed Kaggle notebook version retrieved the repository-scoped `HF_TOKEN` from Kaggle Secrets, authenticated as `tharundevulapalli`, uploaded `storage-tests/kaggle-roundtrip.txt` to the private model repository, downloaded it, and confirmed a matching SHA-256 checksum. The upload produced Hugging Face artifact commit `8748fe1ff0eea0f45e79a47a990d96ae9c6bc086`. The token itself is not stored in this repository. The same restore check must be applied to the first real adapter.

## Starting stack (provisional)

- **Model:** [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Training:** PyTorch, Hugging Face Transformers, PEFT/LoRA, Accelerate, TRL, Datasets.
- **Pilot data:** seeded fictional author/book/prize relationships (proposal suggests ~120 entities: 30 forget, 90 retain).
- **Unlearning baseline:** Gradient Difference, balancing forget-answer loss against retain-answer loss.
- **Editor:** VS Code; **compute:** Kaggle GPU as primary, Colab GPU as fallback.
- **Budget:** aim for zero paid LLM API calls.

These are working choices, not final benchmark settings. Pin dependency versions after the first successful smoke test.

### M0 smoke-test evidence

The interactive and fresh committed M0 runs passed on 2026-11-04 using a Kaggle T4 x2 runtime. The committed run loaded `Qwen/Qwen3-0.6B` at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`, applied the tokenizer chat template in non-thinking mode and produced the response `The model loaded successfully.` No training occurred. See `experiments/runs/m0_smoke_20261104/` for both structured results. The M0 environment/model-loading stage gate is complete.

## What counts as success?

At minimum, M1 must learn the facts, M2 must suppress designated targets without destroying useful ability, and a later unprotected fine-tuning run must provide an honest recovery measurement. Then compare RAFG/RASP with tuned baselines on seven planned metrics: direct/multi-hop forget accuracy, visible reasoning-trace leakage (where available), recovery rate, retain accuracy, new-task reasoning accuracy and computational overhead.

The main comparison is **forgetting stability versus useful new-task learning versus cost**, not suppression alone. All conclusions must be supported by logged tests. Observed suppression does not certify deletion from weights.

## Dataset design

The synthetic relational dataset is specified in [`docs/data/DATASET_SPEC.md`](docs/data/DATASET_SPEC.md). The full seed-42 candidate is saved in [`data/candidates/pilot-v1/seed-42/`](data/candidates/pilot-v1/seed-42/) with 30 F, 90 R and 60 N_rel knowledge units. Independent generations matched byte-for-byte, automated checks reported zero errors, and no near-duplicate names/titles/prizes were found at the configured threshold. The original M0 audit is preserved and verified; remaining manual review and evaluation-protocol decisions precede dataset acceptance.

The candidate contains 840 M1 training prompts, 420 prompts in each later-task condition and 2,520 final prompts. The pinned tokenizer measured 12,480 N_rel tokens versus 12,840 N_reason tokens (2.88% difference, within the 5% volume tolerance). See the [full validation record](experiments/runs/data_full_validation_20261105/README.md). Trainer batching, loss masking and step budgets must still be defined.

CPU-only implementation checks:

```bash
python3 -m unittest discover -s tests -v

python3 -m src.data.generate \
  --config configs/data/pilot_v1.json \
  --output-dir /tmp/persistent-unlearning-tiny \
  --tiny

python3 -m src.data.validate /tmp/persistent-unlearning-tiny

python3 -m src.data.token_budget \
  /tmp/persistent-unlearning-tiny \
  --config configs/data/pilot_v1.json \
  --output /tmp/persistent-unlearning-token-budget.json
```

The tokenizer audit requires `requirements/data-audit.txt`. Automatic normalized-surface similarity checks reject author/book/prize pairs at or above `0.92`. Use the saved candidate for the next audit and preserve its checksums.

## What to do next

1. Review [`context/STATUS.md`](context/STATUS.md) and [`context/DECISIONS.md`](context/DECISIONS.md).
2. Review the [verified utility-prompt comparison](experiments/runs/m1_utility_prompt_diagnostic_20261105/README.md). Original responses reproduce, and the arithmetic-specific instruction removes base abstention on these probes. The one-answer base/adapter strict-score difference is a correct number with units; both make the same three numerical errors.
3. Review the [Qwen3-1.7B comparison protocol](docs/data/QWEN17_CAPACITY_COMPARISON.md). After authorized publication, run [`09_m1_tiny_qwen17_comparison.ipynb`](experiments/notebooks/09_m1_tiny_qwen17_comparison.ipynb) on Kaggle: separate disposable GPU fit probe, then fresh own-model M0 baseline and same 504-row/120-attempt trial. Preserve fit/private adapter/evidence ZIPs; larger size does not guarantee a pass.
4. Build M0 → M1 supervised fine-tuning and verify learned facts.
5. Apply initial machine unlearning to produce canonical M2 and verify forget/retain balance.
6. Run undefended recovery, then begin independent RAFG/RASP branches.

## Continuity and storage

Code/configs belong in version control. Save large model checkpoints/adapters and per-run artifacts in persistent personal storage; free notebook runtimes are temporary. Record base-model revision, adapter lineage, dataset checksums, configs and seed. Restore the repository and critical checkpoints on the personal Dell before returning the office MacBook. Follow office-device policies and never include employer data or credentials.

## Source documents

The proposal PDFs in `docs/proposals/` contain the full idea explanations, proposed formulas, implementation plans and references. Treat those documents as research proposals, not as proof that the methods work.
