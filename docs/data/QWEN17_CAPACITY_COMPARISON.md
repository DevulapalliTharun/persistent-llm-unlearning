# Qwen3-1.7B tiny M1 comparison v1

Status: implemented and tokenizer-prepared locally; no GPU fit, 1.7B baseline inference or training result yet. This compares candidate model checkpoints under a fixed tiny recipe, not an automatic switch of the final dissertation model.

## Fixed identities and controls

| Role | Model | Immutable revision |
|---|---|---|
| Preserved control | Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca |
| New candidate | Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |

The [official 1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B) describes the same thinking/non-thinking interfaces. Public HF metadata/config/tokenizer at the immutable revision were retrieved and verified without model-weight download; metadata is in `experiments/runs/qwen17_metadata_20261106/`. The preserved 0.6B control is the notebook08 120-update run, with config/summary hashes fixed in `configs/training/m1_tiny_qwen17_v1.json`.

Use exactly the same twelve records, 504 augmented targets, 84 development questions, original-training probes, utility questions, LoRA rank/target policy, LR/batches/accumulation, 120 attempts, decoding and numeric gates. Reuse the same trainer through an explicit new run type; old model configs/expected responses remain unchanged. Different model/checkpoint weights, LoRA parameter counts and compute mean this is not a pure size-only causal test or compute-matched benchmark. One seed does not establish general superiority.

## Separate GPU fit probe

Before the real trial, run `--mode fit` in its own subprocess. Load the new pinned model and a temporary fresh LoRA; exercise four accumulated microbatches of the longest actual training rows, AMP/backward/clipping and one provisional AdamW update so optimizer states are allocated. Then exercise representative batch16 sampled thinking generation up to the same 1,024-token cap, reporting actual generated lengths and measured peak memory.

This probe saves no adapter or optimizer state and reports no learning accuracy. Its one temporary update is explicitly recorded and discarded when that process exits. The actual trial starts another process from pristine pinned HF base weights/new LoRA, so its own budget remains120 updates. The probe is representative, not proof that every worst-case generation fits; generation peak can retain temporary training/optimizer buffers. A CUDA OOM or skipped provisional update blocks progression and preserves the fit report/ZIP. Do not silently reduce batches, change precision/quantization or alter the comparison. Non-OOM setup exceptions also need inspection, not a claim of model incapacity.

## Own M0 baseline and training

The 1.7B model must measure its own M0 development/utility outputs before real adapter construction. Do not require it to reproduce 0.6B raw answers, and do not reuse 0.6B baseline accuracy as its starting score. The new config explicitly removes old-model utility reference paths; the baseline guard requires its measured utility >=80% and no caps/missing thinking boundaries. Bare exact values keep their existing format flags. Preserve accidental factual matches and year guesses; do not regenerate/filter data to force zero.

This is a tiny-model selection baseline, not the full 1,260-row candidate knowledge audit. If 1.7B is selected, perform its own full candidate audit, tokenizer-budget/release checks and scaled M1 validation. Keep all 0.6B artifacts/evidence. Never load a 0.6B adapter into the 1.7B pipeline or combine checkpoints across models. Future M2/baselines/RAFG/RASP must share identical canonical starting weights within the selected model.

If the new utility baseline passes, train fresh with the unchanged 120-attempt recipe and save adapter/optimizer/tokenizer/hash manifest before post evaluation. Reuse the exact fact/utility/format/cap/boundary/update/reload gates. Evaluate base178/post346/reload346 predictions, including diagnostic original-training probes:870 total. Save the new own-model baseline, all scored responses/traces/tokens, provenance, exposures, memory/version metadata and gates. Export `model_comparison.json` from the pinned 0.6B control summary and measured 1.7B outputs; do not infer unmeasured improvements.

## Local checks and evidence

All 52 unit tests pass: same training/dev data, strict model pin, unchanged recipe/gates, rejection of cross-model response reuse, and independent baseline acceptance/rejection are covered. Actual pinned 1.7B tokenizer preparation verifies all504 prefixes/masking with no truncation: max106 answer-only/max138 thinking; prompt22503/21495 and response1803/6894 tokens, matching this corpus's 0.6B counts. Evidence: `experiments/runs/m1_tiny_qwen17_preflight_20261106/`. No weights, PyTorch or GPU loaded locally; fit/runtime compatibility and learning remain unverified.

## Execution

Publish source/config/public metadata/control evidence after user authorization, then open `experiments/notebooks/09_m1_tiny_qwen17_comparison.ipynb` on the existing Kaggle GPU kernel with HF_TOKEN. Run sequentially: dependencies/private access, tokenizer preparation, disposable fit, fresh real-trial baseline/training, summary/comparison, private hash-verified backup and evidence ZIP. The fit cell creates a separate small ZIP even on failure; successful fit metadata is copied into the real-trial evidence. Private path `diagnostics/m1-tiny-qwen17-v1/<unique-run-name>/` in the existing private repository. No base weights in Git or evidence ZIP.

Local model-free plan:

```bash
python3 -m src.training.mixed_m1 --config configs/training/m1_tiny_qwen17_v1.json \
  --output-dir /tmp/qwen17-plan-unique --mode plan
```

`--mode prepare` is tokenizer-only; `--mode fit` needs CUDA and discards its temporary update; `--mode train` starts the actual120-attempt trial. Larger size does not guarantee a pass. Select the next model based on measured learning/utility/compute and subsequent recovery feasibility, not the model name alone.
