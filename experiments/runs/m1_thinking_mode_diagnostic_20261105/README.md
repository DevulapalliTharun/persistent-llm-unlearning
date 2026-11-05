# Verified thinking-mode diagnostic — 2026-11-05

All six code cells in notebook 05 completed from commit `3c77324a4c70f987074786ec3810230a3fb53bac`. All fourteen original evidence files are preserved unchanged; all 312 input/score records and all three greedy reproduction controls verify locally. This was inference only, zero updates; original tiny M1 remains failed and no canonical M1/M2 was produced.

| Evaluation | Greedy reference | Sampled non-thinking | Sampled thinking |
|---|---:|---:|---:|
| Base arithmetic | 7/10 | 7/10 | 10/10 |
| Adapter arithmetic | 6/10 | 6/10 | 10/10 |
| Adapter all development | 73/84 | 73/84 | 21/84 |
| F direct author/book | 8/8 | 8/8 | 2/8 |
| R direct author/book | 16/16 | 16/16 | 3/16 |
| F composite | 11/12 | 11/12 | 3/12 |
| R composite | 16/24 | 18/24 | 6/24 |

Counts use the unchanged answer parser. Base thinking returns three correct bare integers without the `Answer:` label, so its 10/10 answer matches coexist with three logged format violations. All twenty thinking utility answers have complete boundaries and no caps. Adapter thinking development has one unclosed/capped response at 1,024 tokens and one corresponding format violation; all other groups have zero caps/boundary failures. Do not equate answer matches with format compliance.

## What the comparison establishes

The matched sampled arms use the same temperature 0.6, top-p 0.95, top-k 20, min-p 0, question ordering, seed scheme and 1,024-token cap. Thinking on/off is the principal changed condition. Greedy controls retain the earlier 128-token budget and reproduce exact original outputs, including all 84 development records.

Thinking fixes the three earlier numerical mistakes on these ten arithmetic questions for both checkpoints in this one sampled run. The inspected arithmetic traces show multiplication of boxes/pens and completion of both operations in the double/subtract and chair examples. This does not establish broad reasoning/utility preservation across tasks or seeds.

Fictional-fact performance changes in the opposite direction: 73/84 versus 21/84. Of the 84 paired answers, twenty are correct in both, 53 lose correctness, one gains correctness and ten are wrong in both. There are 63 thinking failures: one capped/unclosed response and 62 completed responses rejected by the final-answer rule. Two completed year responses contain the correct year plus extra text. Other examples include changed book titles, mixed author names, incorrect prize/year associations and substitutions such as Award for the canonical Prize name. All 84 final answers and selected traces were inspected; selected full traces and all final-answer pairs are in `manual_review.json`.

For example, non-thinking correctly recalls Neris Voss's *The Zephyr Nexus*, while thinking returns *The Zephyr Passage*. Another trace guesses Elira Voss as author of *The Zephyr Nexus* while the final answer says Arven Sable; the expected author is Neris Voss. Generated explanations can contain incorrect statements and contradict their own final answers. The capped response repeats Meteor without producing a final answer.

The adapter was trained with non-thinking prompts, so changed training/evaluation context is a plausible cause of weak thinking-mode fact access. This diagnostic does not isolate that mechanism. No weights were updated; the behavioural difference is not evidence that this evaluation unlearned or erased the facts. Visible traces are generated text, not complete internal computation or a post-unlearning leakage metric. The sampled non-thinking 18/24 R composite draw does not replace the original predefined greedy 16/24 gate failure.

## Evidence and limits

Remote output: `/kaggle/working/m1-thinking-mode-ua9h7f7s`. Archive `m1-thinking-mode-ua9h7f7s-evidence.zip`, SHA-256 `e638d0b8f3a3775953375074ba32087ddf53edadd2218ef80e1489e37b3359c6`. Exact ZIP entries/CRC, original bytes/hashes, all prediction/config/source hashes, raw greedy reproduction, all scores/group summaries, sampled prompt/seed conditions, environment versions and private adapter-hash receipt pass local checks. Saved token accounting is checked on 228 rows against recorded EOS/think metadata; the 84 original greedy development rows lack token IDs. Model inference and tokenizer decode/render were not replayed locally. See `artifact_verification.json` and `verify_artifacts.py`.

Reported duration is 271.4286 seconds on Tesla T4. Stack matches the earlier runs. Private adapter remains at revision `07d23d8b904ab48401f9c6676d05bb85a14d9fbd`; no new adapter/checkpoint was created or uploaded. Recheck from any directory with `python3 /path/to/repo/experiments/runs/m1_thinking_mode_diagnostic_20261105/verify_artifacts.py`.

Next: design a separately versioned tiny M1 training/evaluation retry that tests fictional-fact access in the intended reasoning mode while preserving arithmetic utility. Keep original thresholds/evidence, evaluation arithmetic out of SFT and final-test isolation. No new model size, training hyperparameter, threshold or additional GPU experiment was chosen during this verification. Do not scale to canonical M1 or start M2 yet.
