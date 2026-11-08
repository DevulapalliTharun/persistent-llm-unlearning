# M1 inverse-support v2 preflight — 2026-11-09

CPU/model-free and tokenizer-only evidence, **not GPU training results**. Config/protocol: `configs/training/m1_full_qwen17_inverse_v2.json`, `docs/data/FULL_M1_INVERSE_SUPPORT_V2.md`; notebook15 unexecuted.

`plan.json` freezes5760 rows (original5040 unchanged+720 prize-to-book auxiliary),same840 dev rows,1372 attempts/343-686-1029-1372 checkpoints and original gates/source comparison. Nominal exposure3.811111 passes vs3.809524 before,relative difference0.0417%,not equal compute/tokens. `selected_train.jsonl`/`selected_dev.jsonl` and `augmentation_audit.json` reproduce across model-free and actual-tokenizer preparation. Original dataset/config/model controls remain unchanged.

`masking_report.json` validates5760 prefixes/prompt masks/first EOS with the exact cached pinned1.7B tokenizer. Max107/141 below256; prompt255198/243678 and supervised-response20388/75405 tokens. `token_comparison.json` verifies original5040 encodings/totals still equal frozen v1 and separates auxiliary additions. No model weights/PyTorch/GPU loaded locally.

Source HEAD records the existing published baseline e74fdaa plus exact new working-source hashes, not publication of this new trial. GPU fit/training/1372 applied updates/reload/private round-trips and improved learning are untested. Publish with authorization before Kaggle; preserve source failed1200 model and unchanged dev gate.
