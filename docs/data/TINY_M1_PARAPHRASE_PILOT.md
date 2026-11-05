# Tiny M1 training-paraphrase retry v1

Status: implemented/local tokenizer preparation passed; no GPU training or learning results yet. Notebook `08_m1_tiny_paraphrase_learning.ipynb` is a fresh-base training experiment, not canonical M1. Preserve the original and mixed-mode runs and their failed full acceptance gates.

## Evidence and change fixed before execution

The verified wording diagnostic shows exact training questions 80/84 in each mode, versus development 78/84 answer-only and 63/84 thinking. Seventeen thinking pairs work only with training wording, and four fail both. This motivates testing more training wording variety while tracking residual recall failures; it does not establish a single causal mechanism or guarantee success.

Keep the same pinned Qwen base/revision, twelve entities (4 F/8 R), canonical facts, seven tasks, original training questions, target answers and constructed fact-chain targets. Add two newly authored training-only question templates for every task and entity, without choosing only previously failed items. Counts: 84 original +168 additional questions =252 distinct training questions, each in two modes =504 rows (252 answer-only/252 thinking). Training partition counts remain balanced by the same F/R proportions: 168 F/336 R rows.

The fourteen new templates are in `configs/data/tiny_m1_train_paraphrases_v1.json`, with a pinned SHA in `configs/training/m1_tiny_paraphrase_v1.json`. They include source-to-field lookups and natural-language catalogue/path questions. Require exactly the task's source placeholder (author, book or prize), never its answer entity; reject answer-revealing questions, duplicate questions/IDs and any development-question equality. Reject reuse or near-duplication (SequenceMatcher >=0.92) of any existing template, including reserved monitor/unlearning/final templates. Existing template strings are consulted for negative exclusion/audit only, never model generation/final-score tuning. Manual inspection finds the source/target direction and requested answer type coherent; maximum measured similarity to existing templates is 0.5679012346. This heuristic does not prove linguistic independence.

Keep the original candidate files, original 84 development questions, all final prompts, later-task data and old adapters unchanged. The augmentation is an experiment-specific overlay with its own bank/config/source hashes and saved selected rows/audit. Do not borrow templates allocated to another partition or copy evaluated development questions into SFT. The original 168 targets appear unchanged within the expanded view, with additional provenance fields.

## Training budget and limitations

Start from fresh pinned base/new LoRA, not the existing adapter. Retain every original mixed-mode optimizer/LoRA setting and gate: seed42, rank8/alpha16/all-linear/no dropout, batch4/accumulation4, LR0.0003, clip1.0, max sequence256 and 120 attempts. Do not automatically extend training or select checkpoints by final tests. Same 1,920 planned example presentations are spread across more surface forms; each individual form has fewer repeats than the 168-row run. Supervised response content per semantic question is unchanged, but token lengths/order/exposure and compute are not exactly matched. Record actual per-mode examples and tokens using the existing trainer. This is a feasibility retry, not a token/compute-matched causal ablation or post-unlearning contribution.

Reuse `mixed_m1.py` via an explicit new run type and `paraphrase_data.py` overlay. Original run type/config retain their 168-row plan and 534 predictions. Legacy selectors, encoding, masking, loss, scoring and gate logic are preserved. The new run type validates all original constraints through the existing plan validator.

## Preparation and checks

Actual cached pinned-tokenizer preparation passes 504 prefix/masking checks with no truncation. Answer-only: max106, prompt22503/response1803 tokens. Thinking: max138, prompt21495/response6894 tokens. Response supervision is exactly three copies of the original per-mode totals; thinking targets include the constructed training rationale. See `experiments/runs/m1_tiny_paraphrase_preflight_20261106/`. These are preparation measurements, not model learning results.

All 47 local tests pass, including unchanged legacy plan/targets, full coverage, bank SHA, source-only placeholders, reserved-template rejection and unchanged optimizer/gate constraints. No model/PyTorch/GPU loaded locally.

## Evaluation, gates and storage

Keep the same base utility/reference guard, original development protocols and accuracy/utility/format/cap/boundary/update/adapter-change gates as mixed v1. Base thinking utility must reproduce the recorded reference and be measurable before LoRA construction. Evaluate development non-thinking greedily with128 tokens; development/utility thinking at temperature0.6/top-p0.95/top-k20/min-p0, cap1024, batch16, seed42 plus batch offset. Both fact modes must pass independently. No gate lowering or retrospective mode selection.

At post and reload, additionally generate answers to all 84 **original** training questions in both modes. These recall probes are diagnostic only and are explicitly labelled training scores, never held-out accuracy. They are included in exact reload comparisons, not the acceptance accuracy criteria. Counts: base178 + post346 + reload346 =870 predictions. The additional 168 train-probe questions per stage avoid needing a separate recall run if development still fails. They do not change training data/weights.

Save all original artifacts, new selected targets/audit, before/after/reload outputs, training-wording probe outputs, traces/tokens/termination flags, training logs, hashes, metadata and gates. Save adapter/optimizer/manifest before long post evaluation, then back up privately and verify immutable revision hashes even if gates fail. Private path `diagnostics/m1-tiny-paraphrase-v1/<unique-run-name>/`; no model binaries in Git/evidence ZIP. A successful tiny run still requires full-candidate release review and scaled M1 validation before M2.

## Execution

Publish the new source/config/bank/notebook after user authorization, then run notebook08 on the existing Kaggle GPU kernel with HF_TOKEN. It stops in its first cell if the new source is missing, and stops before training on a failed base utility/reference gate. Use fresh output directories, preserve failed evidence and download the final ZIP after private persistence.

Local plan:

```bash
python3 -m src.training.mixed_m1 --config configs/training/m1_tiny_paraphrase_v1.json \
  --output-dir /tmp/paraphrase-m1-plan-unique --mode plan
```

For cached-tokenizer preparation use `--mode prepare --local-files-only`. Actual CUDA training is `--mode train` through notebook08; it has not been run yet.
