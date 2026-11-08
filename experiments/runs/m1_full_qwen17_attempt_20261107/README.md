# Verified recovered full M1 attempt — 2026-11-07

**Evidence recovery and training execution succeeded. The full-M1 candidate fails one prospective learning gate and is not canonical M1.** Original scores/thresholds/checkpoints remain unchanged.

Notebook12 completed from `35b961e914a57a0fc09cd4c9d02e634670778d73`, fresh `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, reviewed v1.1,30 F+90 R units,5040 mixed/paraphrased targets. Notebook13 recovered the missed private-backup evidence without training/model-weight downloads.

| Suite | M0 | M1 endpoint | Reload |
|---|---:|---:|---:|
| Development, non-thinking | 7/840 | 795/840 | 795/840 |
| Development, thinking | 0/840 | 786/840 | 786/840 |
| Thinking arithmetic probes | 10/10 | 10/10 | 10/10 |

Endpoint accuracy is94.6% non-thinking/93.6% thinking. All post/reload fact/utility outputs have zero format/cap/boundary failures; LoRA changed, all1200 updates applied, exact post/reload records/tokens agree apart from utility batch duration, four periodic/private endpoint receipts verify against saved metadata. These checks do not override the failed learning criterion.

**Sole failed gate:** thinking F prize-to-author21/30 (70%), below75% (at least23/30 required). Non-thinking F prize-to-author27/30 passes. Thinking R prize-to-author69/90 passes but is comparatively weak. All family gates and the other27 per-task gates pass. Nine wrong thinking F responses contain substantive wrong author/association values, not parser/cap errors.

All45 non-thinking and54 thinking wrong final answers, nine focused F reverse-chain traces in full, paired direct/non-thinking outputs and20 base/post utility finals were reviewed. Eight of the nine indirect failures are correct without thinking;8/9 direct book-to-author and9/9 book-to-prize answers are correct with thinking. Eight wrong traces omit the canonical book and sometimes combine other units' associations. This suggests reverse-chain access/wording/composition merits diagnosis, not a proven cause or broad incapacity. No expected-target mentions occur in the99 wrong outputs' visible traces. All correct thinking traces were not manually read. See `manual_review.json`.

Recovered archive `m1-full-hpcajqfx-restored-lnl0x9a4-evidence.zip`, SHA256 `5edb95abef48d756537b6a0ec924c9207d73a1ee46b861e21f40dc2585adf74b`, passes CRC/exact40 entries. All40 bytes are preserved:38 snapshot text/metadata files, separately committed final receipt and recovery provenance. `recovery.json` checks39 recovered-file hashes; `local_verification.json` records independent checks. `reported_result.json` remains the historical pre-recovery snapshot, superseded by verified raw evidence.

Private repo `tharundevulapalli/persistent-llm-unlearning-checkpoints`, path `shared/m1-full-qwen17-v1/m1-full-qwen17-v1-hpcajqfx`, immutable endpoint `a98bc31bcb3f7c404ea91ead35689301ab0c40d3`. Final receipt fetched at `afae2c35412c5656964848c2fdce1016b01ad8f5`. Adapter SHA256 `674aa28ee0bfd806829447844e8eb50c23f768ff1233cfc702b82a5542d12a99` matches the saved attempt1200 checkpoint. Four snapshots/receipts exist at300/600/900/1200. Binaries remain private; local checks rely on recorded runtime binary round-trips, not local binary/inference replay.

Verification reproduces5102 saved prediction records (3422 new/1680 reused M0),5040 canonical targets/840 dev inputs, source/config/input/review/prediction/recovery hashes, all metrics/full verdict,1200 optimizer records, checkpoint-prefix example counts and exact reload. Actual example exposures9596 answer-only/9604 thinking,total19200. Token exposure totals remain runtime-reported, not independent Torch loader/RNG replay. Optional cached pinned-tokenizer replay checks every prediction prompt/text/ID/termination and all5040 training masks/EOS without model weights.

Run `python3 experiments/runs/m1_full_qwen17_attempt_20261107/verify_artifacts.py`; add `--tokenizer-cache /path/to/cache` for offline tokenizer/masking replay. Both checks passed locally.

Training wall2835.6819 seconds includes backup; real-run total3487.5124 seconds (~58.1 minutes), peak allocated6.2464 GiB on T4. Separate117.1324-second disposable probe/one discarded update is not an extra full-training update. UTC November6 timestamps map to November7 IST. Torch2.11.0+cu130/Transformers5.20.2/PEFT0.21.2/Accelerate1.15.0 remain pinned.

**Next proposed work:** inference-only original-training versus saved dev wording for prize-to-author across all F/R/modes on this same endpoint, with reproduction controls. Train-wording recall is diagnostic, not a replacement acceptance metric. No new diagnostic/training/budget, checkpoint selection, gate changes, M2, final-test generation, commit or push occurred here. Earlier tiny pass/all baselines remain preserved.
