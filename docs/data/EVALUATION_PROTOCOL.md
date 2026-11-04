# Pilot answer-evaluation protocol v2

**Version:** `answer-only-v2`

**State:** Implemented and locally tested on 2026-11-05. Revised GPU baseline pending. This protocol is defined from development-audit observations before M1 training or final-test evaluation.

## Why this version exists

The original baseline is preserved under `experiments/runs/m0_knowledge_audit_20261105/`. It used a 32-token cap and whole-response exact matching. Manual inspection found one correct year written in a sentence and several responses ending mid-sentence. Those are evaluation limitations, not evidence of prior knowledge or successful forgetting.

V2 introduces a fixed answer instruction and a larger generation budget, preserves the original scoring as a separate diagnostic, and logs when generation stops at its limit. Results from the two protocols must be reported separately.

## Fixed input and generation settings

- Use `src/evaluation/answers.py` for prompt rendering and answer scoring.
- Send the same system instruction at every compared model checkpoint:

  ```text
  Answer the question using one line: Answer: <answer>. Replace <answer> with only the requested author name, book title, prize name, year, or integer. If you do not know, reply Answer: Unknown. Do not explain, repeat the question, or list alternatives.
  ```

- Keep the canonical question text and dataset files unchanged.
- The M0 v2 audit uses the original 1,260 non-final questions, the same pinned Qwen model/tokenizer revision, seed 42, batch size 16, non-thinking mode and greedy decoding.
- Increase the cap to 128 new tokens. Record an end-of-sequence versus token-limit reason for every response. An EOS appearing at the last allowed token is a normal EOS stop; batch padding after another row finishes does not count as generated content.
- FP16 on T4, with the actual runtime and precision recorded. V2 uses the original audit's native-capability-aware precision selection.

The same versioned wrapper, decoding, precision policy, limits and parser must be used for M0/M1/M2 comparisons. If a later protocol change is needed, version it and establish a comparable baseline; do not tune it against final-test outcomes.

## Answer scoring

The parser does not receive gold answers when extracting a response. Expected answers are used only for the final equality check.

1. Preserve the raw response. Separate any completed visible think block from the final answer; an unfinished block is a parse error.
2. Require a single nonempty line for automatic parsing.
3. If the line starts with `Answer:`, extract the entire following value. Normalize Unicode, whitespace, case, surrounding quotes/Markdown markers and terminal sentence punctuation.
4. Compare the entire extracted value with the normalized expected answer. Preserve articles, names, internal punctuation and digits.
5. A bare single-line answer can still match, but record a missing-label format violation.
6. Keep word-boundary answer mentions as review flags. Do not turn negation, multiple alternatives, a substring inside a longer name, or prose containing the expected string into automatic correct answers.
7. Report year matches separately because matching a year can occur by chance. `Unknown` is an allowed abstention and is incorrect for questions with known target answers.

| Response | Expected | Automatic answer match | Format |
|---|---|---|---|
| `Answer: Mira Venn` | `Mira Venn` | Yes | Compliant |
| `Mira Venn` | `Mira Venn` | Yes | Missing label |
| `Answer: **2023.**` | `2023` | Yes | Compliant |
| `Answer: Not Mira Venn` | `Mira Venn` | No | Has label; answer differs |
| `Answer: Mira Venn or Galen Elden` | `Mira Venn` | No | Has label; answer differs |
| `The prize was given in 2023.` | `2023` | No; review any mention | Missing label/prose |

This is a controlled answer-field metric, not a general semantic judge. Manual semantic findings must be reported separately and never silently inserted into the automated metric. A matching year does not certify a correct supporting reasoning chain.

## Output fields and review

Each v2 prediction adds `extracted_answer`, `answer_match`, `format_compliant`, `parse_error`, `scoring_version`, `generated_token_count`, `eos_generated`, `hit_token_limit` and `stop_reason`. Token counts include the first EOS, when present, and exclude subsequent shared-batch padding.

Use `summary.json` → `answer_scoring` for v2 answer counts, year/non-year counts, format violations, cap hits and per-partition/task results. The root `exact_matches` field remains the original whole-response diagnostic and will reject the new `Answer:` label; it is not the v2 primary metric.

Always retain all predictions, including format failures and capped responses. Cap/format problems must be reviewed; they must not be silently dropped from the denominator or reported as evidence of forgetting. `execution_complete` establishes that the run finished, not dataset acceptance or evaluation adequacy.

## Training compatibility

When the SFT renderer is implemented, use the same system instruction and `training_answer()` to produce assistant targets such as `Answer: The Zephyr Nexus`. Canonical dataset answers stay unchanged. Verify prompt/answer token boundaries and loss masking with the pinned tokenizer before training; those training checks remain pending.

The existing N_rel/N_reason token-volume report predates this wrapper. Recompute training token volumes after the actual SFT/continuation renderer and loss masking are implemented; do not treat the old totals as post-renderer measurements.

## Run the revised baseline

After committing and pushing the new files, open `experiments/notebooks/02_m0_answer_only_audit.ipynb` in VS Code and select a Kaggle GPU kernel. It clones the repository, installs pinned packages, verifies the candidate, runs inference, prints `answer_scoring` and creates a downloadable ZIP. The public base model requires no private HF token.

CPU-only preflight:

```bash
python3 -m src.m0_knowledge_audit \
  --config configs/evaluation/m0_knowledge_audit_v2.json \
  --output-dir /tmp/m0-answer-only-plan \
  --plan-only
```

Use a fresh output directory for each run. Preserve the v2 artifact files in a separate run folder after GPU execution. Full manual collision review, dataset acceptance, M1 learning thresholds and all training remain pending.
