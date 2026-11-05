# Verified tiny training-paraphrase retry — 2026-11-06

All eight notebook 08 code cells completed from commit `c6f53fe0475b220c8eab1a29dd93336db3507661`. All 120 updates applied, adapter changed, post/reload outputs reproduce and the private backup receipt records matching adapter hashes. All twenty-five original small reports and 870 predictions verify locally. **Overall pilot remains failed: the thinking F composite gate is the only false check.** Original/mixed baselines are preserved; this is not canonical M1/M2.

| Post-training suite | Correct |
|---|---:|
| Development, answer-only | 77/84 |
| Development, thinking | 70/84 |
| Original training questions, answer-only | 77/84 |
| Original training questions, thinking | 76/84 |
| Thinking arithmetic utility | 9/10 (base 10/10) |

| Learning gate | Answer-only | Thinking | Minimum |
|---|---:|---:|---|
| F author/book | 8/8 | 8/8 | 90% |
| R author/book | 16/16 | 16/16 | 90% |
| F composite | 9/12 | **6/12 — fail** | 75% (9/12) |
| R composite | 22/24 | 19/24 | 75% |

Every post/reload suite has zero format violations, cap hits and missing thinking boundaries. Utility drops by exactly the allowed ten percentage points; it passes the unchanged small-probe gate, not a broad preservation benchmark. The incorrect chair answer is `5` rather than `15`, with generated reasoning `20 - 8 + 3 = 5`; retain the true 9/10 result.

## Failure review and tradeoffs

All seven answer-only development failures, fourteen thinking development failures, seven/eight original-training failures and the utility error were inspected. Six failed F composite answers consist of three prize-to-author and three author-to-year questions. They include wrong names, wrong years and a prize name returned for a year question. These are completed wrong answers, not a parsing/cap explanation. Thinking F needs three additional correct answers to reach its original gate; no threshold changes or sample/record exclusions.

Compared with the previous mixed adapter, thinking development changes 63/84 to 70/84, but thinking F composite changes 10/12 to 6/12. R composite improves 16/24 to 19/24 and direct answers now all pass. Original-training recall changes 80/84 to 77/84 answer-only and 76/84 thinking; utility changes 10/10 to 9/10. The thinking training/development gap narrows from seventeen to six answers, partly through lower training recall. Do not characterize augmentation as uniformly beneficial or claim a causal contribution from one seed.

The same 120 attempts distribute 1,920 example presentations over 504 rows, so individual surface forms receive fewer repetitions than in the 168-row run. Incomplete training-question generation makes a bounded training-budget experiment reasonable; it does not prove that more steps will fix every error. No new budget, model size, learning rate or threshold was chosen during this verification.

## Verification, artifacts and limits

Remote run `/kaggle/working/m1-tiny-paraphrase-v1-0ar28u5m`; archive `m1-tiny-paraphrase-v1-0ar28u5m-evidence.zip`, SHA-256 `de31fb8f428ffb9f2f02c9a0386425e6e9bc193c0bbaf1ae9b787ad6878a8a8e`. All original entries/CRC/bytes, thirteen prediction hashes, source/config/bank hashes, 504 canonical targets, augmentation audit, unchanged development data and tokenizer-preparation report verify. All 870 scores/stage metrics, 450 saved token records, 120 update/loss/count records and complete failed verdict reproduce. Base utility raw strings reproduce the original reference. Post/reload records match apart from batch duration, including token IDs. See `artifact_verification.json`, `manual_review.json` and `verify_artifacts.py`.

Logged examples: 953 answer-only and 967 thinking (1,920 total). Detailed input/response token totals are runtime reports, not independently replayed data-loader exposures. Private adapter revision `f8fbea97a97cc50f3572c3efb62219dbf5820fa9`, path `diagnostics/m1-tiny-paraphrase-v1/m1-tiny-paraphrase-v1-0ar28u5m`, in the existing private repository. Saved receipt records upload/download hash verification; no local binary download, inference or tokenizer replay during review. The summary's runtime-only persistence string is a pre-upload snapshot superseded by the later receipt.

Reported training time 228.2085 seconds, full run 488.3965 seconds, Tesla T4, peak 3.4322 GiB; no speed-comparison claim. UTC timestamps November 5 correspond to November 6 IST. Recheck with `python3 /path/to/repo/experiments/runs/m1_tiny_paraphrase_attempt_20261106/verify_artifacts.py`.

Next proposed experiment: a separately versioned bounded longer-budget comparison using exactly the same 504 rows and fixed model/LoRA/evaluation/gates. Preserve this 120-step adapter/control; declare budget/checkpoints before execution and report utility/fact tradeoffs. This is a proposal, not an implemented or run experiment. Do not scale to canonical M1 or start M2 yet.
