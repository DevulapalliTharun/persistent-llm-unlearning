# Full M1 preparation evidence — 2026-11-07

This is **CPU/model-free and tokenizer-only evidence, not full M1 training results**. The prepared protocol is in `docs/data/FULL_M1_QWEN17_V1.md`; notebook12 has not run on GPU.

`plan.json` pins reviewed candidate v1.1, corrected M0 evidence, own-model utility reference, recipe/source/config hashes and the1200-attempt endpoint-only policy with four durable checkpoints. `selected_train.jsonl` has5040 rows over all30 F/90 R units, three training-only wordings and both modes; `selected_dev.jsonl` has840 held-out-wording rows. `augmentation_audit.json` records unchanged14 additional templates, isolation and counts. They reproduce between model-free and offline-tokenizer preparation.

`masking_report.json` uses the exact cached public1.7B tokenizer at revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e. All5040 prefixes/masks/first-EOS supervision pass without truncation: max107 answer-only/141 thinking, below256. Prompt totals224898/214818, supervised totals17964/68217. No model weights, PyTorch or GPU inference were loaded locally.

Git HEAD in the preparation plan is the existing published base; new/changed working sources additionally have exact hashes. This does not imply new source is published. GPU memory fit, baseline replay,1200 real updates, periodic/final private round-trips, scaled learning, exact reload and canonical full M1 remain untested/pending. Preserve the earlier passing tiny pilot and corrected M0 baseline.
