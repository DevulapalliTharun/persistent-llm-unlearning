# Tiny M1 mixed-mode retry v1

Status: implemented and locally prepared; no GPU training or learning results yet. This is a separate diagnostic, not canonical M1. Preserve the original answer-only adapter, failed gates and all mode-diagnostic evidence.

## Why this experiment

The original adapter learned facts under non-thinking prompts. The verified thinking-mode run gave both checkpoints 10/10 arithmetic final values, while adapter development fact accuracy changed from 73/84 without thinking to 21/84 with thinking under matched sampling. Training/evaluation mode mismatch is a plausible explanation, not a causally established mechanism. Instead of changing model size or weakening gates, test a fresh tiny adapter trained with both response formats.

## Fixed data and targets

Same pinned Qwen model/revision, unchanged candidate, first four F/eight R units and original 84 `m1_train` questions. Each question has two training rows, total 168:

| Training mode | Assistant target | Count |
|---|---|---:|
| Non-thinking | `Answer: <canonical value>` | 84 |
| Thinking | `<think>` constructed canonical fact chain `</think>` then `Answer: <canonical value>` | 84 |

Chains are deterministically constructed from the selected canonical entities and the question's task path. For example, author-to-year uses author wrote book, book received prize, prize awarded in year. Reverse questions use the corresponding canonical relationship. These are synthetic rationales supplied as supervision, not generated/observed internal reasoning or evidence that the model has reasoned correctly. All seven task families are represented in each mode. No arithmetic, development wording, N or final-test questions are used for training. Development retains the same 84 questions under held-out wording about the same learned facts.

Composite answers/chains are explicitly supervised. Success measures access under held-out wording and modes, not independent generalization to unseen compositions. The ten arithmetic questions have already guided setup diagnosis and are development probes, not an independent final reasoning benchmark; passing them cannot establish broad reasoning preservation.

Use the existing v2 system prompt for fact training/evaluation. Non-thinking rendering is unchanged. Thinking rendering explicitly uses `enable_thinking=True` for both full assistant targets and inference prefixes. Verify exact token-prefix agreement, mask prompt/padding labels, supervise all assistant response tokens plus first EOS, and refuse truncation. Thinking supervision includes the constructed rationale, so report response CE rather than calling it final-answer-only CE.

Cached pinned-tokenizer preparation verified all 168 rows: non-thinking max 101 tokens, 7,153 prompt / 601 supervised response tokens; thinking max 134 tokens, 6,817 prompt / 2,298 supervised response tokens. Prefixes match; zero truncation. Evidence: `experiments/runs/m1_tiny_mixed_preflight_20261105/`. This is preparation evidence only, not training.

## Initialization and budget

Start from fresh pinned Qwen base weights and a newly initialized rank-8 all-linear LoRA, alpha 16, dropout 0, no bias updates. Never load the previous tiny adapter as the training starting point. Keep original seed 42, 256-token sequence limit, batch 4, accumulation 4, AdamW learning rate 0.0003, zero decay, gradient clip 1.0 and 120 optimizer attempts. Base is FP16; only FP32 LoRA parameters train. Record skipped/applied AMP updates, actual examples/input/response tokens seen by mode, per-mode response CE and training time.

The mixed data have twice as many rows and longer responses. Same attempted updates do not imply matched token exposure, equal exposures per fact/mode or equal compute relative to the original. This is a combined representation/mode feasibility experiment, not an isolated causal ablation proving that rationales alone fix the issue. No automatic budget extension if the pilot fails. Final optimizer/scaler state is saved for inspection, not exact mid-run resume.

## Evaluation fixed before execution

At base, trained and freshly reloaded stages:

| Suite | Rows per stage | Generation |
|---|---:|---|
| Fact development, non-thinking | 84 | Original greedy, 128-token cap |
| Fact development, thinking | 84 | Temperature 0.6, top-p 0.95, top-k 20, min-p 0, 1,024-token cap |
| Arithmetic utility, thinking | 10 | Same sampled thinking settings, arithmetic-specific instruction |

Evaluation batch 16 and seed 42 plus batch offset follow notebook 05 exactly. Use a separate evaluation-settings dict so training batch 4 cannot change evaluation batching/seeds. There are 178 predictions per stage, 534 if all stages complete. This is a go/no-go check with declared mode-specific settings, not a causal comparison of thinking to greedy. It is one sampled seed; do not select a best seed or tune against final tests.

Before adapter construction/training, arithmetic must score at least 8/10, reproduce the saved pinned-base thinking response strings exactly, and have no caps/boundary failures. A failed baseline gate stops before LoRA creation or optimizer updates and saves reports. Correct bare values remain accepted with format flags by the unchanged v2 parser; the three previously observed base-label omissions are recorded. This does not change old scores or gates.

For each fact mode separately, keep original per-F/R author/book >=90% and composite >=75% accuracy requirements. Thinking arithmetic post accuracy must drop no more than ten points from its measurable baseline. Retain the original zero post-development/post-utility format/cap requirements; additionally require zero missing thinking boundaries. Preserve all 84 task scores, including supporting one-hop questions outside the original author/book/composite gates. Targets mentioned only inside an unfinished trace cannot pass. Require changed adapters, all 120 applied updates, and exact post/reloaded raw responses in every suite, including sampled traces. A pass is still only a tiny diagnostic pass, not canonical M1.

## Execution and artifacts

Runner `src/training/mixed_m1.py`, target renderer `src/training/mixed_data.py`, fixed config `configs/training/m1_tiny_mixed_v1.json`, notebook `experiments/notebooks/06_m1_tiny_mixed_learning.ipynb`. The notebook clones main, checks Kaggle dependencies/private storage, prepares targets, trains, summarizes, uploads and verifies adapter files, then exports the small evidence ZIP. Publish new files plus the verified notebook 05 reference evidence with user authorization before the clone runs.

Save adapter/tokenizer/optimizer and a hash manifest immediately after the 120 attempts, before long post-training generation. Completed adapter can be backed up even if later evaluation fails. Keep all baseline/post/reloaded responses/traces, rendered sampled prompts/token IDs, training exposures, source/config/data/reference hashes, generation metadata, environment and gate reports. Private path: `diagnostics/m1-tiny-mixed-v1/<unique-run-name>/` in the existing private HF repository. Verify upload/download hashes at the immutable artifact revision. No base weights in Git or the evidence ZIP.

Model-free plan from the repository root:

```bash
python3 -m src.training.mixed_m1 --config configs/training/m1_tiny_mixed_v1.json \
  --output-dir /tmp/mixed-m1-plan-unique --mode plan
```

Tokenizer-only preparation uses `--mode prepare`; add `--local-files-only` for cached tokenizer preparation. `--mode train` runs CUDA training through notebook 06. Always use a new empty output directory. Independently verify saved metrics/private receipt after the run before choosing any next stage.
