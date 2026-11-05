# Verified training/development wording diagnostic — 2026-11-06

All seven notebook 07 code cells completed, using published helper commit `7d4b28f1d18d543ea0daacf8f09cb2df9dbf0eed` and the unchanged saved mixed-mode adapter. All fourteen original evidence files and all 346 input/score records verify locally. Original development/utility raw outputs reproduce exactly. **No training or weight updates occurred; the mixed-mode pilot remains failed and canonical M1/M2 are not produced.**

| Mode | Exact training questions | Development wording | Utility |
|---|---:|---:|---:|
| Answer-only | 80/84 | 78/84 | Not rerun in this mode |
| Thinking | 80/84 | 63/84 | 10/10 |

These are the same 84 record/task/answer pairs with different question wording, not new facts in development. Both training-question groups have zero format/cap/boundary failures. Development thinking retains the original one capped/unclosed response/format violation; utility has no violations or caps.

## What the paired answers show

Thinking has 63 correct with both wordings, seventeen correct only on training wording, zero correct only on development wording, and four wrong with both. Thus most additional development failures in this run are associated with the wording change. Training generation is strong but imperfect; this is not proof that all facts are memorized or that wording is the only cause. Sampling uses one fixed seed.

Answer-only has 77 correct both, three training-only correct, one development-only correct and three wrong both, a smaller net gap. The four training errors differ somewhat across modes. Answer-only errors include two prize-to-year, one author-to-year and one prize-to-author; thinking errors include two author-to-year, one prize-to-year and one prize-to-author. All eight training failure records and all discordant pair identities/questions were inspected. Thinking failures include wrong 1997 versus 1996 for Neris Voss, 1996 versus 2006 for Lunar Pine Prize, Pella Farrow versus Quin Weyr for Umber Wing Prize, and 1951 versus 1952 for Pella Farrow's award year. These are wrong associations/values, not merely unit/label formatting.

The seventeen thinking training-only matches cover year, reverse-lookup and supporting relation tasks. For example, asking `In which year was Young Meteor Prize awarded?` gives the correct 1996, while `Name the award year for Young Meteor Prize.` gives 1971 in the reproduced development output. This supports a wording-generalization limitation under the declared setup. It does not establish a general model mechanism or guarantee that paraphrase augmentation will fix every error. Training-question accuracy is diagnostic and cannot replace held-out performance, justify gate lowering or promote the original failed pilot.

## Verification and provenance

Archive `m1-wording-probe-hlq0680r-evidence.zip`, SHA-256 `9048b324c3a18fcf8167304e32bb3983b519943bcb9a70b43452b394ff11225c`; remote output `/kaggle/working/m1-wording-probe-hlq0680r`. ZIP entries/CRC/bytes, config snapshot/canonical hash, original training config, pinned helper source hashes, 84 exact pairings, six original reference hashes, adapter manifest receipt, all 346 scores/five group metrics, 178 saved token records and all 168 paired final-response/category records verify. The three original raw controls reproduce exactly. See `artifact_verification.json`, `manual_review.json` and `verify_artifacts.py`.

Same private adapter at revision `25764276e3e83e6c20519429854e15216f57e676`, path `diagnostics/m1-tiny-mixed-v1/m1-tiny-mixed-v1-_avmkde0`. Remote restore/hash verification is recorded; no binary download/model/tokenizer replay occurred locally during review. Model generation was sampled only in thinking suites, with the unchanged seed/cap/instruction settings. Reported evaluation-section time 139.9381 seconds on Tesla T4 excludes model/download preparation. UTC timestamps are November 5; local IST execution is November 6. No speed comparison or new model upload.

Recheck with `python3 /path/to/repo/experiments/runs/m1_training_wording_diagnostic_20261106/verify_artifacts.py` from any directory.

Next proposed concrete training correction: a separately versioned fresh-base mixed-mode tiny retry with additional new training-only paraphrases and a declared exposure budget. Preserve canonical candidate, all prior adapters, development/final wording and original thresholds. Do not copy evaluated development questions into SFT or ignore the four exact-training generation errors. This correction has not been implemented or run in this review; we remain at M0 → M1.
