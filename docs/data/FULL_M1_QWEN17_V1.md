# Prospective full M1 candidate — Qwen3-1.7B v1

Prepared2026-11-07. **No full-M1 GPU fit/training/evaluation or new private checkpoint upload has run.** Config: `configs/training/m1_full_qwen17_v1.json`; runner: `src/training/full_m1.py`; notebook: `experiments/notebooks/12_m1_full_qwen17_learning.ipynb`.

## Fixed starting point and data

Start fresh from public `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` with a new LoRA. Never continue the tiny adapter. Use the reviewed checksum-pinned candidate v1.1, all30 F+90 R units, and the unchanged train-only paraphrase bank from the passing tiny pilot. The scoped data review is a prerequisite for protocol preparation, not automatic training authorization; execution is explicitly requested by the notebook's training cell after the prospective protocol is prepared/published.

| View | Scope |
|---|---|
| Original SFT questions | 840, seven tasks per F/R unit |
| Wording variants | Original plus two training-only paraphrases per task |
| Unique SFT questions | 2520 |
| Assistant formats | Answer-only and supplied canonical fact chain + answer |
| SFT rows | 5040 =2520 each mode;1260 F/3780 R |
| Development | 840 fixed held-out-wording questions, same learned units |
| Later N_rel/N_reason data | Not trained or post-evaluated in this M1 run |
| Final tests | Static exclusion checks only; no model input/scoring/tuning |

Composite answers/chains are explicitly supervised. Development tests wording/mode access to learned facts, not independent generalization to previously untrained compositions. Supplied rationales and visible traces are not complete evidence of hidden reasoning. Preserve thinking as intended primary setting and non-thinking as a separate same-checkpoint access check. Existing tiny configs/results and their gates remain unchanged.

## Prospective training and checkpoint selection

- FP16 unquantized base, FP32 trainable LoRA only; rank8,alpha16,dropout0,all-linear targets,bias none.
- Same masked per-example mean response CE as passing pilot; thinking supervision includes supplied rationale/answer/first EOS. Max sequence256, reject truncation.
- AdamW LR0.0003,weight decay0; same optimizer defaults, clip1.0, AMP scaler initial1024.
- Microbatch4,accumulation4,seed42;5040 rows are divisible by the batch size.
- **1200 attempted updates**, requiring all1200 to apply. This is10x tiny data/attempts,19200 nominal example exposures and about3.8095 dataset passes. It preserves nominal per-row exposure rather than guaranteeing scaled learning, equal token exposure or a causal comparison. Actual per-mode token/example exposures are logged.
- Save/back up at **attempts300/600/900/1200**. Only endpoint1200 can qualify. No intermediate dev evaluations, best-dev selection, early stopping, silent budget extension, alternate mode or precision/batch fallback. Skipped updates fail the budget gate.
- Each checkpoint contains LoRA/tokenizer/optimizer/scaler, attempt/update/config metadata, the flushed training prefix, masks and source/config plan. Private immutable upload/download SHA256 round-trips must verify adapter, optimizer and tokenizer files before training continues. Failure stops the attempt and preserves local/previous durable evidence.
- Checkpoints are recoverable artifacts, but **exact mid-run resume is not implemented**; loader/RNG cursor replay is not claimed. No automatic restart/retry after failed controls, upload failure or OOM. Preserve the failed attempt for review.

Private path: `tharundevulapalli/persistent-llm-unlearning-checkpoints/shared/m1-full-qwen17-v1/<unique-run>/checkpoints/attempt-<six digits>`. Endpoint/reports get a separate final folder round-trip in the notebook. No full base weights or binaries enter Git/the evidence ZIP. Periodic backup time is included in reported training wall time; receipt timing isolates upload/round-trip work but does not imply a pure GPU-time comparison.

## M0 baseline and GPU execution

The verified corrected baseline already contains all840 F/R development rows in both modes. That F/R subset has7 answer-only year matches and0 thinking matches, distinct from the full180-unit audit's9/1 counts that include N_rel; no non-year matches in either view. Pin its original13 archive files/local verification, scoped review and the tiny1.7B own-M0 utility reference. Preserve its chance-compatible year matches, high format/cap counts and inherited provenance rather than forcing a zero baseline.

Before actual training, run a separate-process disposable memory probe with four longest real training rows/four accumulated backwards/one temporary update and representative batch16 thinking generation. No saved probe adapter or learning result. Validate exact config/source hashes and longest-token coverage before the actual process starts fresh; probe update is not counted as a full-M1 update. This is representative fit, not a worst-case OOM guarantee.

Actual training checks the original T4/package versions, tokenizer/generation metadata and private repository. Generate32 fresh fact controls (first16 fixed dev rows in both modes) and10 thinking utility answers. Require exact output/token/score replay against the own-model references, ignoring only batch-duration/origin metadata; require measurable utility>=80% and no utility caps/boundary failures. A mismatch stops **before LoRA creation or real updates**, with evidence. If matched, copy1680 M0 dev records with explicit reuse provenance, not another full M0 inference run.

At the endpoint, evaluate all840 dev questions in each mode and ten thinking arithmetic probes, then repeat after attaching the saved adapter to a fresh pinned base:1690 outputs per stage,3380 endpoint/reload outputs. Total planned new evaluation answers3422 including42 base controls/utility;1680 reused M0 records are separate. Original-training wording recall probes are omitted at full scale because they were diagnostic only; this does not alter dev gates or earlier results.

Facts keep the exact v2 system/parser, independent840-row ordering,batch16,seed42+offset,greedy non-thinking cap128 and thinking sampling temperature0.6/top-p0.95/top-k20/min-p0,cap1024. Utility uses the existing arithmetic-specific instruction. Every new fact response saves prompt/raw text/token IDs/final-vs-reasoning fields/EOS/caps/format/boundaries/mentions. Final correctness excludes trace-only mentions. Post SFT reasoning mentions are not post-unlearning leakage results.

## Full candidate gates declared before GPU results

Keep original family gates in both F/R and modes: author↔book>=90%, aggregate composite>=75%, base utility>=80%,post utility drop<=10 percentage points, zero post fact/utility format/cap/boundary failures, all1200 updates, changed LoRA and exact save/reload text/tokens/fields apart from batch duration.

Add **per-task** full-data requirements: each one-hop task>=90% and each multi-hop task>=75%, independently in F and R in each mode. This includes book-to-prize/prize-to-year attributes that were absent from the tiny family's main threshold buckets. Denominators are30 F or90 R per task/mode; require full840-row input coverage/order before scoring. An aggregate score cannot hide a failed task. Require all four periodic private round-trip receipts. These are new prospective full-candidate criteria; they do not retroactively change the passed tiny verdict.

The ten reused arithmetic probes are small development diagnostics, not independent broad utility evidence. Passing these gates produces a **reviewable full-M1 candidate**, with `canonical_m1=false` until independent result/private-persistence review. It establishes neither M2 forgetting nor persistent-unlearning benefits. Preserve failed gates; revise a later experiment explicitly rather than lowering thresholds after outcomes.

## Local verification and next action

All79 local tests pass, including shared tiny-target compatibility, full data isolation, fixed budget/mode/storage policy, per-task failures hidden by aggregate scores, coverage/cap/reload/update failures, and mocked private artifact round-trips/corruption refusal. The passing tiny verifier and corrected-baseline verifier still pass. These tests do not execute GPU training or real HF uploads.

Actual offline pinned-tokenizer preparation verifies5040 targets, all prefixes/prompt masks/EOS and no truncation:

| Mode | Rows | Longest sequence | Masked prompt tokens | Supervised response/EOS tokens |
|---|---:|---:|---:|---:|
| Answer-only | 2520 | 107 | 224898 | 17964 |
| Thinking | 2520 | 141 | 214818 | 68217 |

Evidence: `experiments/runs/m1_full_qwen17_preflight_20261107/`. No torch/weights/GPU loaded locally. Shared changes are minimal: optional augmentation unit count defaults to12, optional train-step checkpoint callback defaults toNone, and progress displays the configured budget. Legacy selected targets/gates/recipes remain identical.

```bash
python3 -m src.training.full_m1 --config configs/training/m1_full_qwen17_v1.json \
  --output-dir /tmp/full-m1-plan-fresh --mode plan
# Tokenizer preparation only, no model weights:
python3 -m src.training.full_m1 --config configs/training/m1_full_qwen17_v1.json \
  --output-dir /tmp/full-m1-prepare-fresh --mode prepare
```

Publish the prepared source/config/tests/protocol/preflight and verified corrected-baseline evidence with authorization. Run notebook12 sequentially on the existing Kaggle T4/HF_TOKEN, complete final backup, then download the text evidence ZIP. Inspect real gates/errors/update/token/checkpoint receipts before canonical promotion or M2 design.
