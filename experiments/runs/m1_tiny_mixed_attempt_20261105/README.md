# Verified mixed-mode tiny M1 retry — 2026-11-05

All eight notebook 06 code cells completed from commit `7d4b28f1d18d543ea0daacf8f09cb2df9dbf0eed`. Training applied all 120 updates; adapter changed, fresh-base reload reproduced all post-training responses/token IDs, and private adapter upload/download hashes were verified by the saved receipt. All twenty original small reports are preserved and checked. **The answer-only side passes its declared gates, but the thinking side fails; overall mixed-mode pilot remains failed.** This is not canonical M1/M2.

| Evaluation | Base | Post training | Reloaded |
|---|---:|---:|---:|
| Fact development, answer-only | 0/84 | 78/84 | 78/84 |
| Fact development, thinking | 0/84 | 63/84 | 63/84 |
| Arithmetic, thinking | 10/10 | 10/10 | 10/10 |

| Post-training learning gate | Answer-only | Thinking | Minimum |
|---|---:|---:|---|
| F author/book | 8/8 | 7/8 | 90% |
| R author/book | 16/16 | 15/16 | 90% |
| F composite | 10/12 | 10/12 | 75% |
| R composite | 23/24 | 16/24 | 75% |

Thinking F direct is 87.5%, below 90%; R composite needs at least 18/24. Thinking also has one capped/unclosed F prize-to-year response and corresponding format violation. All ten post utility outputs follow the required label and are correct; utility baseline exactly reproduces the saved thinking reference. Base utility has the same three correct bare values/format flags as before. No threshold was lowered or result replaced.

## Failure review and interpretation

All six answer-only failures are wrong associations/years: two author-to-year, three prize-to-year and one prize-to-author. Thinking has 21 failures: six author-to-year, eight prize-to-year, four prize-to-author, two book-to-author and one book-to-prize. Twenty have complete traces/final answers; one repeats a prize-to-prize chain to the 1,024-token cap without a final answer. One completed response includes the correct year with extra prize text; the strict score remains unchanged. All completed failed thinking traces/finals and the capped tail were inspected, as were all post arithmetic finals. Full records are in `manual_review.json`.

The constructed-chain training now produces short fact-chain-style traces, but some chains contain incorrect years, authors or invalid relations. Example: Neris Voss's book/prize are correct in one trace, followed by a wrong 2000 year rather than 1996. Another prize-to-author query returns a book title, and several prize-to-year traces say one prize received another prize. Low training response CE does not guarantee generalization to held-out question wording.

Thinking development is 63/84 versus the prior adapter's 21/84 under the declared thinking settings. This is a descriptive diagnostic comparison: doubled rows, different mode exposures and longer supervision prevent attributing the change to one isolated cause or claiming a method result. Arithmetic is preserved on ten repeatedly inspected development probes, not a general benchmark. Composite answers/chains are explicitly supervised, so this is not proof of unseen-composition reasoning. Original answer-only pilot remains independently preserved and failed.

## Reproducibility and storage

Archive `m1-tiny-mixed-v1-_avmkde0-evidence.zip`, SHA-256 `4eabb6f0b6374cafddbf29ac37cb3b0fa53f250d9af0db8e71216de8d94b1632`; remote run `/kaggle/working/m1-tiny-mixed-v1-_avmkde0`. All original ZIP bytes/entries/CRC, nine prediction hashes, config/source provenance, 168 canonical targets, prior tokenizer masking report, 534 prediction scores, 282 saved token records, stage metrics and full failed verdict pass independent checks. All 120 updates and 1,920 logged examples agree: 962 answer-only / 958 thinking. Reported input/response tokens are 88,791/6,883 answer-only and 103,897/26,163 thinking; these detailed exposure totals were not independently replayed through a data loader. Post/reload records are identical apart from `batch_seconds`, including token IDs.

Private repository: `tharundevulapalli/persistent-llm-unlearning-checkpoints`; artifact revision `25764276e3e83e6c20519429854e15216f57e676`; path `diagnostics/m1-tiny-mixed-v1/m1-tiny-mixed-v1-_avmkde0`. The saved persistence receipt records adapter-file round-trip verification. Summary's runtime-only persistence string is a pre-upload snapshot; later receipt records durable backup. Adapter/scaler/optimizer/tokenizer binaries/settings stay private; no local binary download or model inference replay occurred during review. Reported training time 229.4284 seconds; full run 607.7479 seconds, Tesla T4, peak 3.3516 GiB. No speed-comparison claim.

Recheck from any directory with `python3 /path/to/repo/experiments/runs/m1_tiny_mixed_attempt_20261105/verify_artifacts.py`. See `artifact_verification.json` for limits.

Next proposed diagnosis: compare training-wording generation with the saved development-wording results on this same adapter, with modes/seeds fixed before execution. This tests whether failures concern basic training-prompt recall or wording/mode generalization before changing updates, model size or templates. Notebook 07 is now completed and verified in `../m1_training_wording_diagnostic_20261106/`: training wording 80/84 in both modes versus development 78/84 and 63/84. The next proposed correction is training-only paraphrase diversity with held-out prompts/gates intact. No selective record exclusions or canonical M1/M2 promotion.
