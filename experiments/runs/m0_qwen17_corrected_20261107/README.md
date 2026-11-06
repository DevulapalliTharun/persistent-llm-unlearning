# Verified corrected Qwen3-1.7B M0 baseline — 2026-11-07

All six notebook11 code cells completed from `ed86cbaaae144a706dd2f2be3cee9d0cca13db10`. **All64 replay controls match every saved field/token exactly.** The full affected eight-row batch was then regenerated in each mode:16 refreshed answers. This is a derived baseline with **2520 records =16 refreshed +2504 unchanged verified parent records**, plus64 separately saved controls, not a fresh2520-answer GPU run. Every derived row records its origin and parent-summary hash.

Fresh public `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP16 Tesla T4; no adapter, optimizer, training or weight updates. Candidate v1.1 changes only ku120's book title plus version metadata; all IDs/splits/other facts/tiny12 train/dev rows are preserved. Final-test questions were not model-evaluated.

| Derived baseline | Answer-only | Thinking |
|---|---:|---:|
| Questions | 1260 | 1260 |
| Final-answer matches | 9 | 1 |
| Non-year matches | 0 | 0 |
| Format violations | 809 | 751 |
| Token-limit hits | 0 | 16 |
| Missing thinking boundaries | 0 | 15 |
| Expected-answer reasoning mentions | 0 | 4 |

All final matches/reasoning mentions remain year values. All13 potential-answer flags are inherited exactly from the parent baseline apart from origin metadata; their [prior manual interpretations](../m0_qwen17_full_20261106/manual_review.json) still apply. The high protocol-failure counts and limited observations do not establish absent internal knowledge or demonstrate the intended canonical chains.

All16 refreshed outputs and eight full thinking traces were reviewed:0/8 matches in each mode, no expected final/trace mentions, caps or missing boundaries. Only book-to-author/book-to-prize generated outputs change in each mode; the other six responses per mode reproduce. For the corrected title, book-to-author gives John Christopher answer-only and Robert Jordan thinking, rather than canonical Brel Lume. These are noncanonical responses, not evidence that the synthetic author-book association was already known. Thinking format violations decrease752→751 because its book-to-author answer now includes the requested label; factual accuracy does not increase. See `manual_review.json`.

Archive `m0-qwen17-corrected-aqe8clfe-evidence.zip`, SHA256 `3cd1062dd3fa9c3a42a46665487706de8a3c5348da4f6e1b25bb2306a4988ca8`, contains13 original text files preserved byte-for-byte here. CRC/exact entries/hashes pass. `local_verification.json` records local checks; original Kaggle `artifact_verification.json` and `summary.json` are unchanged. Runtime183.6535 seconds (~3.1 minutes), peak allocated5.0173 GiB; CUDA/package versions match parent. UTC November6 timestamps correspond to November7 IST.

Independent verification reconstructs all2520 rows/provenance and grouped summaries, checks64 exact controls, rescores80 new records, verifies input/config/source/review/file hashes and all review selections. Optional cached-tokenizer replay checks2600 saved record occurrences (80 controls/refreshed +2520 derived, with intentional overlap), including prompts/decoded text/termination metadata. No model weights or inference were replayed locally. Controls sample reproducibility rather than proving all parent batches would regenerate identically.

```bash
python3 experiments/runs/m0_qwen17_corrected_20261107/verify_artifacts.py
# Optional offline exact-tokenizer replay:
python3 experiments/runs/m0_qwen17_corrected_20261107/verify_artifacts.py \
  --tokenizer-cache /path/to/cache
```

**Review decision:** `data_review.json` records that v1.1 passes the bounded controlled-pilot data review for prospective full-M1 protocol preparation. Confirmed complete-title ambiguity is version-corrected; weaker cross-domain/partial-name overlaps remain documented limitations, not a claim of universal surface uniqueness. All original year matches/failures/candidates are retained. The automatic summary's `dataset_accepted=false` remains untouched; this separate scoped review does not authorize full training automatically.

**Next:** declare full M1 update/checkpoint budget, data/token exposure and development gates before training30 F+90 R from a fresh pinned public1.7B base. Preserve the successful mixed-mode target representation/paraphrases, thinking as intended main setting, separate non-thinking checks and final-test isolation. No full M1/M2, unlearning or recovery result is claimed. No source/data changes, commit or push occurred in this result review.
