# Thinking-mode diagnostic v1

Status: notebook 05 completed on Kaggle from `3c77324`; all fourteen original reports and all 312 predictions independently verified. This was inference only on existing checkpoints, before more SFT or unlearning. See `experiments/runs/m1_thinking_mode_diagnostic_20261105/`.

The utility instruction diagnostic removed base abstention but left the same three numerical mistakes in both checkpoints. Tiny M1 also failed R composite development accuracy. Test the model's supported thinking mode before attributing these errors to model capacity or choosing another training configuration.

## Comparison fixed before execution

| Arm | Thinking | Decoding | Output cap | Purpose |
|---|---|---|---:|---|
| Greedy reference | Off | Original greedy | 128 | Reproduce saved predictions |
| Sampled control | Off | Temperature 0.6, top-p 0.95, top-k 20, min-p 0 | 1,024 | Control for sampling and output budget |
| Sampled thinking | On | Same sampling settings | 1,024 | Compare thinking at matched sampling settings |

The [official pinned Qwen model card](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md) documents `enable_thinking` and warns against greedy decoding in thinking mode. Use its recommended thinking sampling parameters in both sampled arms; keep the original greedy arm separate. The card recommends substantially longer outputs for general reasoning; our 1,024-token cap is a resource-bounded feasibility choice, not a maximum-capability benchmark. Cap hits and unfinished reasoning must be visible in reports, never silently treated as ordinary wrong final answers or retried selectively.

Evaluate the original ten arithmetic questions on base and adapter (60 answers across three arms), plus all 84 original F/R development questions on the adapter (252 answers), total 312. There are 104 sampled-thinking responses. Do not select only previously failed examples. Base relational questions are not rerun in this bounded check; base arithmetic is the reasoning/utility reference. Same model revision, FP16, verified private adapter, questions and suite-specific system instructions as the earlier checks. Arithmetic uses `ARITHMETIC_PROMPT`; relational development uses the original v2 instruction. Each sampled arm uses seed 42 plus batch offset, batch size 16 and the same example ordering; it is one sampled run, not a multi-seed estimate. No training, N examples or final-test prompts.

## Final answers and visible reasoning

Save the generated text, rendered prompt, token IDs, EOS/pad IDs, template hash, batch seed/time, visible reasoning text, final answer and termination details. Score the final answer with the unchanged answer-only-v2 parser. In thinking mode, require exactly one complete `<think>...</think>` boundary before accepting a final answer; a target appearing only inside reasoning does not count. Support an opening marker already supplied by a tokenizer prompt, and record missing/malformed boundaries explicitly. Correct numbers with units remain strict mismatches. Keep boundary failures, cap hits and strict-answer failures separately countable, even where they overlap.

Visible generated explanations may be useful to diagnose a skipped arithmetic step or incorrect relational association; they are not a complete view of internal computation or independent proof of genuine composition. This is M0/M1-stage learning diagnosis, not post-unlearning trace leakage. The adapter was trained with thinking disabled, so mode-specific degradation may reflect that training/evaluation mismatch; no guarantee that thinking helps.

## Reproducibility and interpretation

Config: `configs/evaluation/thinking_mode_diagnostic_v1.json`; runner: `src/evaluation/mode_diagnostic.py`; notebook: `experiments/notebooks/05_m1_thinking_mode_diagnostic.ipynb`. Pins include original development/reference hashes and the adapter's existing private HF revision. Exact greedy arithmetic strings must match notebook 04 references, and exact greedy relational strings must match original post-training development records. A mismatch requires reproduction diagnosis before broader interpretation.

Compare sampled thinking to sampled non-thinking within each checkpoint/suite to assess mode sensitivity. Comparing thinking directly to greedy mixes mode, sampling and output-budget changes. Report all outcomes, including capped/unclosed responses and utility/development tradeoffs, without choosing the best sample. A single-seed improvement would motivate further validation, not promote the original failed pilot or establish persistent unlearning. Original thresholds/scoring remain unchanged; canonical M1/M2, initial unlearning and recovery are still pending.

## Running

The new source/config and verified utility reference files must be available on GitHub main before the notebook clones it. Commit/push only with user authorization. Use the existing Kaggle GPU runtime, attach HF_TOKEN through Kaggle Secrets, run cells sequentially, and export/download the evidence ZIP. The notebook restores the existing adapter; it does not upload models or train.

Model-free local plan from the repository root:

```bash
python3 -m src.evaluation.mode_diagnostic \
  --config configs/evaluation/thinking_mode_diagnostic_v1.json \
  --output-dir /tmp/thinking-mode-plan-unique --mode plan
```

## Verified outcome — 2026-11-05

All three greedy controls reproduce exactly. Matched sampled utility scores change base 7/10 to 10/10 and adapter 6/10 to 10/10 when thinking is enabled. Three base thinking finals omit the Answer: label; their values are accepted by the unchanged parser with recorded format violations. All twenty thinking utility responses complete without caps.

Matched adapter development accuracy changes from 73/84 non-thinking to 21/84 thinking. The paired counts are twenty correct in both, 53 losses, one gain and ten wrong in both. There is one 1,024-token cap with an unfinished boundary; 62 other thinking failures have complete boundaries. All final answers and selected visible traces were inspected, revealing mixed/invented facts and some trace/final contradictions, not just formatting issues. Non-thinking sampled R composite is 18/24, while thinking is 6/24; the original greedy 16/24 gate remains unchanged.

This one-seed run demonstrates opposite arithmetic/fact-access behaviour by mode. The original non-thinking SFT versus thinking evaluation mismatch is a plausible explanation that needs a controlled training comparison; no causal training mechanism or broad model-capacity limit is established. No unlearning or weight update occurred. Next design a separately versioned, mode-consistent tiny M1 training/evaluation retry before scaling or starting M2; no new training setting/threshold was selected in this verification.
