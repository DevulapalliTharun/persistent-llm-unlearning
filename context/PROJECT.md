# Project Context — Persistent LLM Unlearning

**Project type:** M.Tech major project  
**Status:** Two proposed approaches; experiments not yet established  
**Last reviewed:** 2026-11-07

## 1. Research problem

Approximate machine unlearning can reduce an LLM's ability to produce designated information, but **later legitimate fine-tuning can sometimes make that information recoverable again** (benign relearning). We want forgetting to remain effective **without unnecessarily harming the model's ability to learn new reasoning tasks**.

This project studies **targeted fictional factual associations** that can be accessed through direct questions or multi-hop reasoning. It does **not** aim to erase arithmetic or reasoning ability as a whole, and it does not claim certified deletion from all model weights.

## 2. Two candidate solutions

### A. RAFG — Reasoning-Aware Forgetting Guard

**Reactive / on-demand protection.** During downstream reasoning fine-tuning, periodically evaluate a small monitoring set. If evidence suggests forgotten answers are becoming more likely, briefly activate a corrective forgetting loss. Otherwise continue ordinary fine-tuning.

- Monitor answer-token cross-entropy (CE) relative to the post-unlearning checkpoint M2.
- Keep monitoring prompts, corrective-training prompts and final tests distinct.
- Compare against no guard, always-on/fixed guard and periodic guard, including monitoring overhead.
- Proposed research question: Can recovery-triggered correction maintain forgetting with fewer unnecessary updates or lower total cost than strong fixed/periodic controls?

Proposed (not yet validated) sketch:

```text
risk_j(t) = max(0, CE_j(M2) - CE_j(t) - delta)
L_guard = mean_j max(0, margin_j - CE_j(t))
L_total = L_new + gamma * active * L_guard
```

`active` is 1 only during triggered correction steps. Teacher-forced CE is a monitoring proxy; actual generated answers still need evaluation.

### B. RASP — Retain-Aware Selective Plasticity

**Preventive / selective protection.** At M2, estimate which LoRA modules appear sensitive to forget targets relative to retain targets. During later fine-tuning, penalize changes to higher-risk modules more strongly, allowing other modules to adapt.

- Candidate heuristic: clipped/normalized ratio of mean forget-gradient magnitude to mean retain-gradient magnitude, per module.
- Compare against no protection, uniform anchoring, random protection and forget-only rankings.
- Proposed research question: Can retain-aware selective regularization reduce target recovery at comparable new-task learning performance?

Proposed (not yet validated) sketch:

```text
g_F,i = mean parameter-wise gradient magnitude for module i on F
g_R,i = mean parameter-wise gradient magnitude for module i on R
score_i = normalize_and_clip(g_F,i / (g_R,i + eps))
L_RASP = L_new + lambda * sum_i w_i * ||U_i(t) - U_i(M2)||^2
```

`U_i` is the effective scaled LoRA update (typically based on B @ A). The gradient ratio is a heuristic, not proof of a causal recovery circuit.

## 3. Common pipeline — do once, then branch

```text
M0: open-weight base model
  → synthetic-knowledge SFT on all fictional facts
M1: fact-trained model
  → existing Gradient Difference unlearning on F while preserving R
M2: post-unlearning model (validated and saved)
  → ordinary later-task fine-tuning from an M2 copy
  → demonstrate/quantify possible recovery without protection
  ↳ RAFG experiments: monitor + conditionally correct
  ↳ RASP experiments: score modules + selectively regularize
```

**Non-negotiable experimental rule:** Save one canonical M2 model state and start every unprotected, baseline and defended continuation from **the identical effective M2 weights/adapters**. Do not redo initial SFT/unlearning separately for RAFG and RASP. Each branch should use the same planned new-task data, evaluation splits and checkpoints where possible.

If ordinary downstream fine-tuning does not show meaningful recovery in either planned condition, there is no established recovery problem for a protection method to solve in that setup. Investigate and report this result rather than assuming recovery.

## 4. Data design (provisional pilot)

Start with **fictional author–book–prize–year relationships** generated using a fixed seed. Suggested initial scale from the proposals: **120 entities, 30 forget and 90 retain**, adjustable after a smoke test.

- **F — forget:** selected fictional author–book associations to suppress.
- **R — retain:** other associations and useful behaviors to preserve.
- **N — new-task:** legitimate unrelated math/logic or new-entity relational reasoning; must not contain exact forgotten targets.
- **T — held-out:** unseen direct, paraphrased, 2-hop/3-hop and optionally trace-oriented recovery tests; never train or tune on T.
- **RAFG-only:** separate monitor set and corrective guard-training set; neither is the held-out final test set.

Split **entities first**, then generate templates so an entity cannot leak across forget/retain partitions. Ensure multi-hop question wording does not reveal its answer; ensure relationships used for joining entities are unambiguous. Use separate templates for training, module scoring/monitoring, development and final testing.

The implementation-level schema, entity-first hash split, prompt-use matrix, leakage checks, manifest and acceptance gate are defined in `docs/data/DATASET_SPEC.md`. In that specification, `T` is a held-out prompt view rather than a fourth trainable entity partition.

Example: `The Glass Observatory` → author `Mira Venn`; prize `Silver Comet Prize`; ask directly for author or indirectly through the prize/book chain. These associations are invented; surface strings still require collision review.

## 5. Baseline unlearning — reused method, not our novelty

Use supervised fine-tuning to produce M1. For M1 → M2, start with **Gradient Difference**, minimizing:

```text
L_unlearn = -alpha * CE(F) + beta * CE(R)
```

This raises target-answer loss while encouraging retained answers to remain likely. Choose a checkpoint that demonstrably suppresses previously learned targets **and** retains reasonable utility. Do not identify "the model refused once" with permanent erasure.

## 6. Shared evaluation plan

Seven metrics are included in the proposed evaluation plan:

| Metric | Main purpose | When |
|---|---|---|
| Direct Forget Accuracy | Can the model answer direct target questions? | M1, M2, post-SFT |
| Multi-hop Forget Accuracy | Can it recover targets through reasoning chains? | M1, M2, post-SFT |
| Reasoning-Trace Leakage Rate | Do targets appear in available generated reasoning text? | M2, post-SFT, if traces exposed |
| Recovery Rate | Of previously learned targets forgotten at M2, how many reappear? | During later fine-tuning |
| Retain Accuracy | Are non-target fictional facts preserved? | M1, M2, post-SFT |
| New-task Reasoning Accuracy | Does the model learn useful new tasks? | M2 baseline, post-SFT |
| Computational Overhead | Extra training/inference time, passes and memory? | During later fine-tuning |

**For M0 → M2, prioritize metrics 1, 2, 5, plus baseline reasoning ability (6).** Collect trace leakage when supported; compute recovery and protection overhead after M2. Also log answer-token CE as a diagnostic, especially for RAFG.

```text
Recovery@t = (# targets learned at M1 and forgotten at M2 that are recovered at step t)
             / (# targets learned at M1 and judged forgotten at M2)
```

Predefine the M2 forgetting criterion before final evaluation and report the denominator. A visible generated trace is not a complete record of hidden reasoning.

## 7. Practical implementation constraints

- **Model candidates:** `Qwen/Qwen3-0.6B` remains the initial engineering/control model. The separately pinned `Qwen/Qwen3-1.7B` passed the same-recipe tiny pilot and is the working candidate for the next shared-pipeline checks. Full-dataset/final model validation remains pending; each model needs its own M0/M1/M2 baselines and tokenizer checks.
- **Candidate adaptation:** LoRA, rank 8 or 16 for pilot.
- **Working mode policy:** Preserve the passing mixed answer-only/canonical-chain training recipe. Thinking is the intended main research setting; evaluate non-thinking separately on the same checkpoint to detect access/forgetting that differs by mode. Supplied synthetic rationales and visible traces are not a complete account of hidden reasoning.
- **Candidate libraries:** PyTorch, Transformers, PEFT, Accelerate, Datasets, TRL, NumPy, pandas, matplotlib.
- **Development:** VS Code on current office Mac M2 Pro, later personal Dell G15.
- **Compute:** free Colab or Kaggle GPU sessions, if available; code can be edited in VS Code while notebook kernels execute remotely.
- **Budget goal:** no paid model APIs/tokens. Free GPU quotas and availability are not guaranteed.
- **Portability:** store source/configs in Git; back up data manifests, M1/M2 adapters, checkpoint metadata and metrics outside disposable cloud runtimes.
- **Workstation handoff:** migrate/verify the project on the personal Dell before the office laptop is returned on January 29 (year/date subject to the user's actual schedule).
- **Company policy:** use the office machine for personal research only if permitted; keep company data, credentials and infrastructure out of the project.

## 8. Scope and research honesty

The proposed contributions are **post-unlearning safeguards**, not a new original machine unlearning algorithm. Related work already addresses importance weighting, circuit localization, relearning resistance and constraint-based training. Method advantages and novelty claims depend on experiments against strong baselines, matched utility and compute, and preferably multiple seeds.

## 9. Proposal documents

- `docs/proposals/RAFG.pdf` — Reasoning-Aware Forgetting Guard, 23-page research/implementation guide.
- `docs/proposals/RASP_Project_Guide.pdf` — Retain-Aware Selective Plasticity, 24-page research/implementation guide.

These documents are **proposals, not completed experimental results**. Consult them for detailed examples, formulas, implementation sketches, related-work references and limitations.
