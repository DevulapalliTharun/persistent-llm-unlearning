# Full pilot candidate validation — 2026-11-05

The seed-42 candidate contains 30 F, 90 R and 60 N_rel knowledge units. Two separate processes, using Python hash seeds 1 and 97, produced byte-identical files. The saved repository candidate matches both copies and passes checksum verification. See [`summary.json`](summary.json) and [`token_budget.json`](token_budget.json).

The first full generation failed because the arithmetic generator repeated a question. The correction deterministically skips repeated questions and is covered by a 420-row uniqueness/arithmetic-correctness regression test. Nine tests pass.

The exact pinned Qwen tokenizer measured 12,480 total N_rel tokens versus 12,840 N_reason tokens, a 2.884615% difference within the existing 5% tolerance. Both contain 420 rows. This checks text volume; it does not establish equal loss-token counts, measured compute or trained utility. Those controls must be defined in the trainer.

All automated dataset validation checks passed, and no near-duplicate author/book/prize pairs were found at the configured 0.92 threshold. The generator's generic token-audit warning is resolved by the separate detailed report stored here. Full external name/title collision review and the M0 accidental-knowledge audit remain pending.

The full candidate is saved under [`data/candidates/pilot-v1/seed-42/`](../../../data/candidates/pilot-v1/seed-42/). The generator source hashes and config snapshot in its manifest identify the exact working code used, including changes made after the recorded baseline Git commit.

Commands used for the two independent generations (temporary directory names may vary):

```bash
env PYTHONHASHSEED=1 python3 -m src.data.generate \
  --config configs/data/pilot_v1.json --output-dir /tmp/pllu-full-verified.qk622h
env PYTHONHASHSEED=97 python3 -m src.data.generate \
  --config configs/data/pilot_v1.json --output-dir /tmp/pllu-full-reproduced.HLvla6
diff -qr /tmp/pllu-full-verified.qk622h /tmp/pllu-full-reproduced.HLvla6
python3 -m src.data.validate data/candidates/pilot-v1/seed-42
```

Run the tokenizer audit in an environment with `requirements/data-audit.txt` installed:

```bash
python -m src.data.token_budget data/candidates/pilot-v1/seed-42 \
  --config configs/data/pilot_v1.json --output /tmp/full-token-budget.json
```

No model weights were loaded and no training was performed during this CPU validation.
