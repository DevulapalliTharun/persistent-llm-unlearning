# Persistent LLM Unlearning — RAFG & RASP

M.Tech major-project research workspace for studying whether **targeted knowledge unlearning stays effective during later legitimate reasoning fine-tuning**.

**Current state:** Project proposals and common experimental plan. No experimental gains or completed training checkpoints are claimed here.

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
│   └── proposals/
│       ├── RAFG.pdf
│       └── RASP_Project_Guide.pdf
├── src/                # Reusable data/training/evaluation code (future)
├── experiments/        # Experiment configs/notebooks (future)
├── references/         # Notes and related-work references (future)
├── AGENTS.md           # Instructions for coding assistants
├── .gitignore
└── README.md
```

## Starting stack (provisional)

- **Model:** [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B)
- **Training:** PyTorch, Hugging Face Transformers, PEFT/LoRA, Accelerate, TRL, Datasets.
- **Pilot data:** seeded fictional author/book/prize relationships (proposal suggests ~120 entities: 30 forget, 90 retain).
- **Unlearning baseline:** Gradient Difference, balancing forget-answer loss against retain-answer loss.
- **Editor:** VS Code; **compute:** available free Colab/Kaggle GPU, or a permitted local machine.
- **Budget:** aim for zero paid LLM API calls.

These are working choices, not final benchmark settings. Pin dependency versions after the first successful smoke test.

## What counts as success?

At minimum, M1 must learn the facts, M2 must suppress designated targets without destroying useful ability, and a later unprotected fine-tuning run must provide an honest recovery measurement. Then compare RAFG/RASP with tuned baselines on seven planned metrics: direct/multi-hop forget accuracy, visible reasoning-trace leakage (where available), recovery rate, retain accuracy, new-task reasoning accuracy and computational overhead.

The main comparison is **forgetting stability versus useful new-task learning versus cost**, not suppression alone. All conclusions must be supported by logged tests. Observed suppression does not certify deletion from weights.

## What to do next

1. Review [`context/STATUS.md`](context/STATUS.md) and [`context/DECISIONS.md`](context/DECISIONS.md).
2. Create/verify the environment and smoke-test model loading on the chosen GPU.
3. Write a reproducible synthetic data generator and evaluation splits.
4. Build M0 → M1 supervised fine-tuning and verify learned facts.
5. Apply initial machine unlearning to produce canonical M2 and verify forget/retain balance.
6. Run undefended recovery, then begin independent RAFG/RASP branches.

## Continuity and storage

Code/configs belong in version control. Save large model checkpoints/adapters and per-run artifacts in persistent personal storage; free notebook runtimes are temporary. Record base-model revision, adapter lineage, dataset checksums, configs and seed. Restore the repository and critical checkpoints on the personal Dell before returning the office MacBook. Follow office-device policies and never include employer data or credentials.

## Source documents

The proposal PDFs in `docs/proposals/` contain the full idea explanations, proposed formulas, implementation plans and references. Treat those documents as research proposals, not as proof that the methods work.
