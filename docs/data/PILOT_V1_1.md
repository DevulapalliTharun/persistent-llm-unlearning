# Candidate v1.1 and bounded M0 refresh

Completed and independently verified on 2026-11-07: all64 replay controls match,16 affected-batch responses regenerated,2504 parent records reused with provenance. All80 new records and2520 derived rows verify, including offline tokenizer replay; all16 refreshed outputs/eight traces reviewed without new target flags. [Scoped data review](../../experiments/runs/m0_qwen17_corrected_20261107/data_review.json) supports prospective full-M1 protocol preparation, not automatic training authorization. Full-M1 budget/config/training remain pending. Notebook: `experiments/notebooks/11_m0_qwen17_corrected_candidate.ipynb`.

## Single evidence-based correction

The unchanged v1 full M0 audit is complete and preserved in `experiments/runs/m0_qwen17_full_20261106/`. It is not being erased or rerun merely to remove its year matches.

An independent [catalogue record](https://openlibrary.org/works/OL84773W/Silver_Citadel) identifies *The Silver Citadel* under Anthony Horowitz. Candidate `ku_000120` assigns that complete title to Brel Lume. Version v1.1 changes only this canonical book title to **The Veylith Citadel**. Author, prize, year, record/entity IDs, all F/R/N_rel memberships, seed and templates are retained. The version metadata changes on entity rows; their other factual fields remain identical except the declared title.

Quoted searches for the replacement complete title, title plus Brel Lume, and title with book/title context returned unrelated uses of Veylith/citadel rather than an exact complete-title association in the inspected results. These include [unrelated fiction](https://www.royalroad.com/fiction/144074/the-ashes-in-ascendance/chapter/3238850/chapter-52-starforge-aftermath-where-loyalties). This is bounded review, not proof that a surface never existed. Existing weaker shared words, awards with longer names, artwork/site/game/character overlaps remain in the original collision review; they do not establish the invented author-book-prize graph. No other surfaces are changed to optimize M0 scores.

The correction is applied to a checksum-pinned **copy** of v1. Do not call the original entity generator with a different version: its version-dependent split namespace would change F/R membership. Use the dedicated correction command:

```bash
python3 -m src.data.correct_candidate --config configs/data/pilot_v1_1.json \
  --output-dir /tmp/pilot-v1-1-reproduced
python3 -m src.data.validate data/candidates/pilot-v1.1/seed-42
```

Output refuses overwrite and includes correction/source/config provenance, validation and checksums. Two independent outputs match byte-for-byte. All 6,600 relational prompt records retain their original IDs/paths/templates; only the explicit title substitution changes question/answer strings. Final prompts are mechanically updated for consistency only, never model-evaluated. `n_rel_train` and `n_reason_train` are byte-identical to v1, so the already-measured pinned 1.7B preliminary token budget remains applicable. All selected tiny12 training/development rows are unchanged.

## Derived baseline, with exact replay controls

Only three of the 1,260 allowed baseline input rows change: `ku_000120` author-to-book (answer), book-to-author and book-to-prize (questions). All lie in the original final eight-row F/R development batch, offset832. The other five rows must also be regenerated: batch padding and sampled trajectories can depend on batchmates. Never mix old and new answers inside an affected batch.

Keep the same pinned 1.7B weights, T4, FP16, package versions, system prompt, parsing, generation policies, input ordering, batch shapes and per-view seed42+offset. The runner first generates the first unchanged16-row batch in **each** input view and **both** modes: **64 control answers**. Every output field and token ID must match its verified parent record exactly. Any mismatch stops before generating affected batches/merging; raw controls, comparison details and a failed summary are retained. Do not retry a failed control until it happens to pass. Inspect the mismatch and prepare a complete corrected-candidate audit if reuse is unsuitable.

If controls pass, regenerate the complete affected eight-row batch in both modes: **16 answers**. The resulting baseline has **2,520 records = 16 refreshed + 2,504 unchanged parent records**; the64 controls are saved separately. Every merged row carries its origin and parent-summary hash. This is a derived baseline with 80 new GPU outputs, not a new full-run result. Controls sample replay consistency; they do not prove every reused batch would reproduce on another run.

The runner verifies the frozen parent archive-file/config/source/input/review hashes and corrected-candidate checksum/report. It saves new raw IDs/text/prompts, all review subsets, summary, package versions and per-row reuse provenance. Its offline verifier reconstructs the merge, rescores new rows/token metadata, checks controls and every derived row/summary/review selection. It also validates a failed-control run without treating it as complete. No adapter, optimizer, updates, final-test inference or automatic dataset acceptance occurs.

```bash
python3 -m src.evaluation.corrected_m0 \
  --config configs/evaluation/m0_qwen17_corrected_v1_1.json \
  --output-dir /tmp/corrected-m0-plan --mode plan
# GPU execution: --mode audit with a fresh output directory.
# Afterwards, offline evidence reconstruction:
python3 -m src.evaluation.corrected_m0 \
  --config configs/evaluation/m0_qwen17_corrected_v1_1.json \
  --output-dir /path/to/completed/run --mode verify
```

## Next research stage

Notebook11 was published with authorization at `ed86cba` and completed. Its ZIP/raw files, independent verification and all refreshed-output reviews are preserved in `experiments/runs/m0_qwen17_corrected_20261107/`. All13 potential-answer flags are exact inherited records apart from origin metadata. Original full-audit year matches, protocol failures and limitations remain; thinking format violations decrease one without changing accuracy. A separate `data_review.json` records the bounded controlled-pilot review for full-M1 protocol preparation. Raw automatic accepted=false flags remain unchanged; scoped review does not itself authorize full training.

Then prepare the full M1 protocol for **30 F +90 R units**, starting fresh from the pinned public1.7B base rather than continuing the tiny adapter. Preserve the successful mixed-mode target representation and training-only paraphrases. Thinking is the intended main research setting, while non-thinking remains a separate access/forgetting check on the same checkpoint. Training/gates cover both modes; visible synthetic rationales are supplied supervision, not proof of internal reasoning. Declare the scaled update budget, checkpoints and dev criteria before GPU training; no such full-training config or result is claimed here. N_rel/N_reason remain later-task data and final-test prompts stay held out.
