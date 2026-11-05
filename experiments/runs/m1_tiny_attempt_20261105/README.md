# Verified tiny M1 diagnostic — 2026-11-05

**Outcome:** Execution completed, but the predefined pilot gates did not all pass. This is a diagnostic adapter, not canonical M1. All 14 original evidence files from the downloaded ZIP are now preserved and checked. Earlier user-pasted stdout remains in `reported_result.json`, with rounded progress samples in `reported_loss_samples.json`.

## Reported learning and infrastructure

| Development group | Correct / total | Preset minimum | Result |
|---|---:|---:|---|
| F direct author↔book | 8/8 | 90% | Pass |
| F composite tasks | 11/12 | 75% | Pass |
| R direct author↔book | 16/16 | 90% | Pass |
| R composite tasks | 16/24 | 75% | Fail |

The model learned all tested direct author/book answers under held-out development wording. Composite task access was weaker for R. This does not establish independently demonstrated reasoning: composite answer rows are also included in training.

Reported adapter-change, optimizer-budget, format, generation-limit and fresh-base save/reload gates passed. The printed answer CE fell substantially and the last printed value was 0.0007; low training CE does not override the development failures.

The base arithmetic utility score was 0/10 and post-training score was 4/10. The preset measurable-baseline requirement (at least 8/10) failed. Although the nondecrease check is mechanically true, this run cannot establish useful preservation from a meaningful initial utility baseline. Inspect `base_utility.jsonl` and `post_utility.jsonl` for unknown/format/wrong-answer behaviour before proposing a revised probe. Do not infer the failure's cause or weaken the original threshold from the aggregate alone.

The retry ran after the TorchAO setup blocker. The environment freeze confirms torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2 and Accelerate 1.15.0, with TorchAO absent. Training-source hashes match commit `5ec5960`; the current working tree's runner additionally contains the documented early TorchAO check. The saved config and committed source explicitly use non-thinking greedy decoding. No model inference was replayed locally.

## Preserved artifacts and verification

Remote output: `/kaggle/working/m1-tiny-run-8myx_22r`.

The saved Kaggle receipt records successful adapter-file upload/download hash verification in private repository `tharundevulapalli/persistent-llm-unlearning-checkpoints`, path `diagnostics/m1-tiny/m1-tiny-run-8myx_22r`, immutable artifact revision `07d23d8b904ab48401f9c6676d05bb85a14d9fbd`. The adapter/optimizer binaries remain in private storage; local verification covers the downloaded evidence and receipt, not a second binary download.

Archive: `m1-tiny-run-8myx_22r-evidence.zip`, SHA-256 `239998bd1c0eec8685dc8e7fc0cf533c0ca6b09ccb400889ec1045f11b3fd548`. ZIP CRC and all 14 extracted-file hashes passed. Inputs match the unchanged candidate; config/source provenance, all 282 prediction records, all 120 optimizer records, printed loss samples, summary metrics and failed gates reproduce. Post/reloaded development and utility records are identical. See `artifact_verification.json` and `manual_review.json`.

Recheck from any directory with `python3 /path/to/repo/experiments/runs/m1_tiny_attempt_20261105/verify_artifacts.py`. No model packages or GPU are required. Raw token IDs were not exported; EOS/token counts are internally consistent but cannot be replayed. Adapter-change and remote round-trip claims rely on the original runtime report/receipt. The original plan's `model_executed=false` and summary's `artifact_persistence` string are pre-execution/upload snapshots; the completed summary and later receipt record the subsequent stages.

## Raw-response diagnosis and next step

All ten base utility responses are exactly `Answer: Unknown.`. After training, four are strictly correct, three have correct numbers with units (`7 apples`, `18 coins`, `12 kilometers`), and three are wrong (20 instead of 10, 12 instead of 7, and 12 instead of 15). Manual semantic review therefore finds 7/10 numerically correct; the original strict result remains 4/10 and the base gate still fails. Abstention does not establish an absence of arithmetic ability. Wrapper sensitivity is a hypothesis requiring a controlled comparison.

The eight R composite failures comprise five author-to-year and three prize-to-author outputs. Three year outputs contain the correct year plus extra text; two year answers and all three author answers are incorrect. Across all 84 development rows, 73 pass and 11 fail; two failures (one F book-to-prize, one R prize-to-year) are one-hop and outside the direct author/book and composite gates. Zero label-format violations does not imply every answer has the requested type/value.

Next: define a separately versioned utility-prompt diagnostic on the same pinned base and saved adapter before another SFT attempt. Preserve original v2 scores/gates and final-test isolation. No prompt/config/model change or additional training was performed during this verification. Full dataset release and canonical M1/M2 remain pending.
