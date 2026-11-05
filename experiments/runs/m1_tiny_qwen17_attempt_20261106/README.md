# Verified passing Qwen3-1.7B tiny M1 pilot — 2026-11-06

All nine notebook09 code cells completed from commit `4cc4c6c5c6f4754fe2688e5b4bbd08544cc75e81`. **Every preset tiny pilot gate passes**, independently recomputed from all 870 predictions and 120 real optimizer records. Memory probe, exact reload and private adapter backup checks pass. This is the first fully passing recorded tiny M1 pilot, not canonical full M1/M2 or a completed unlearning result.

| Suite | Own 1.7B M0 | Post-training | Preserved 0.6B post |
|---|---:|---:|---:|
| Development, answer-only | 0/84 | 81/84 | 77/84 |
| Development, thinking | 0/84 | 81/84 | 70/84 |
| Original-training probe, answer-only | Not run | 82/84 | 77/84 |
| Original-training probe, thinking | Not run | 79/84 | 76/84 |
| Thinking utility | 10/10 | 10/10 | 9/10 |

| Post learning gate | Answer-only | Thinking | Minimum |
|---|---:|---:|---|
| F direct author/book | 8/8 | 8/8 | 90% |
| R direct author/book | 16/16 | 15/16 | 90% |
| F composite | 12/12 | 11/12 | 75% |
| R composite | 21/24 | 23/24 | 75% |

All post/reload suites have zero format violations, cap hits and missing thinking boundaries. Adapter changed and all 120 updates applied. Post/reload records agree except batch duration, including sampled token IDs. Data and reported per-mode example/token exposures match the 0.6B control; model/LoRA parameter counts and compute differ, so this is one-seed practical feasibility evidence, not a causal size-only or broad superiority claim.

## Remaining errors and baseline caveats

All three development failures in each mode and all two/five original-training failures were inspected. Answer-only development errors are Ulric Islen's award year (1366 versus 1955), an extra Cove in Pella Farrow's prize name, and Cira Rusk instead of Aven Korr for Frost Quill Prize. Thinking development errors are Neris Voss's year (2006 versus 1996), wrong author for *The Luminous Passage*, and wrong author reached from Harbor Ember Prize. These are completed wrong values/associations, not cap or formatting failures. Passing the declared thresholds does not mean perfect recall or independent unseen-composition reasoning; composite answers/rationales were explicitly supervised.

Own-model M0 fact scores are zero under the fixed parser, with 57 non-thinking/52 thinking format violations and one thinking cap. No expected final-answer mentions are flagged in either 84-row base set; reviewed examples invent incorrect book/author associations. Preserve the high base format-violation counts and raw outputs. Zero protocol accuracy is not proof of absent internal knowledge. Full candidate M0 audit/release review remains pending for 1.7B.

## GPU fit, training and storage

Disposable fit succeeded on Tesla T4 (14.5622 GiB device memory), with a temporary one-update adapter discarded/no checkpoint saved. Probe peaks: training 6.1567 GiB, representative generation 5.3302 GiB. Real trial starts fresh in a separate process and applies its own 120 updates; do not count the probe as a 121st update of the real model. Real peak allocated memory 6.1984 GiB. This establishes fit for the tested configuration, not every future workload. Probe reports/plan are copied into the main ZIP, so a separate local fit ZIP is not required for this verification.

Real training reports 8,716,288 trainable parameters, 953 answer-only/967 thinking examples, and the same logged input/response exposure totals as the 0.6B control. Detailed token exposure is runtime-reported, not independently replayed data-loader ordering. Reported training 283.4753 seconds; complete real trial 738.9067 seconds; probe 106.1474 seconds. No speed-comparison claim.

Private repository `tharundevulapalli/persistent-llm-unlearning-checkpoints`, immutable artifact revision `5fa54c9e7108f2261b16a0903666baf34072f407`, path `diagnostics/m1-tiny-qwen17-v1/m1-tiny-qwen17-v1-qv7fs_sw`. Saved receipt records adapter upload/download hash verification. Summary runtime-only persistence string is an earlier snapshot; later receipt records successful backup. No local binary download or model/tokenizer replay during independent review.

## Verification and next step

Archive `m1-tiny-qwen17-v1-qv7fs_sw-evidence.zip`, SHA-256 `01f897adf19ce416b3a705f7f5595ffadeabd2326ecee4758a9ee64e770873c1`; remote run `/kaggle/working/m1-tiny-qwen17-v1-qv7fs_sw`. All 28 original files are preserved unchanged. ZIP CRC/entries/bytes, thirteen prediction hashes, config/model/control/reference/source hashes, 504 targets/audit, tokenizer preparation, all 870 scores/450 token records, full passing verdict, own-baseline guard, update/loss/count arithmetic, reload equivalence, copied fit scope/memory and candidate/control comparison reports verify. Environment versions match the recorded stack. UTC November 5 timestamps correspond to November 6 IST. See `artifact_verification.json`, `manual_review.json` and `verify_artifacts.py`.

Recheck with `python3 /path/to/repo/experiments/runs/m1_tiny_qwen17_attempt_20261106/verify_artifacts.py` from any directory.

Next: treat 1.7B and this passing tiny recipe as the working candidate for shared-pipeline preparation. Finish full-candidate manual acceptance and 1.7B-specific full M0 knowledge/token-budget audits, then predefine the scaled M1 training/evaluation budget. Keep held-out final prompts untouched, retain all 0.6B controls, and do not mix cross-model checkpoints. No full M1/M2, new audit/training, final dissertation model selection or recovery result is claimed here.
