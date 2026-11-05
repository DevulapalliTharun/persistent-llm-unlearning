# Full candidate M0 audit for Qwen3-1.7B

Status: prepared/model-free validated; no full-audit GPU results yet. The passing twelve-unit tiny M1 pilot supports the model/recipe, but neither full-candidate acceptance nor canonical M1/M2 is complete.

## Fixed scope

Fresh `Qwen/Qwen3-1.7B` at immutable revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP16 on one GPU. No adapter loaded, optimizer or weight updates. Keep seed42, batch16, v2 system prompt/parser and the declared generation policies from the passing tiny setup:

| Mode | Generation | Questions |
|---|---|---:|
| Answer-only | Non-thinking greedy, cap128 | 1,260 |
| Thinking | Temperature0.6/top-p0.95/top-k20/min-p0, cap1,024 | 1,260 |

Inputs are unchanged `m1_dev.jsonl` (840 F/R questions over30/90 units) and `n_rel_train.jsonl` (420 questions over60 new units). Total180 units/seven task families/2,520 outputs. N_rel is audited for accidental knowledge, not trained during M1. No N_reason or final-test question is sent to a model. Static final templates/samples may be examined only for leakage exclusion and dataset structural review; no final model scoring or tuning.

Batch the two input views independently and reset seed offsets per view, avoiding a mixed N_rel/F/R final development batch. Future full M1 development evaluation can use the same840-row ordering/batch shapes/seeds. This is a new full-model baseline; do not substitute old0.6B scores or the trained tiny adapter's outputs.

## Outputs and review policy

Save two JSONL files with every original input, rendered prompt, generated token IDs, raw text, visible reasoning/final separation, EOS/cap/format/boundary fields, unchanged parsed-answer scoring and legacy diagnostics. Primary correctness uses final answers only. Expected-answer mentions inside complete or incomplete visible reasoning are separate review flags, never extra correct answers or a post-unlearning leakage metric. Keep chance year matches separate from non-year targets. Repeated/negated alternatives and mere mentions need manual interpretation.

The runner flushes each batch and prints progress. Preserve partial runs on interruption; output overwrite is rejected and no automatic selective resume/target retry is implemented. Completed summaries include per-mode/partition/task counts, source/config/input/review hashes, generation metadata, versions/memory/time and explicit zero-training/no-adapter flags. Dataset acceptance remains false even when execution completes.

Review every potential correct answer/final mention/reasoning mention in `potential_answers_for_review.json`, alongside context. Inspect collision-flagged rows and representative protocol failures/caps across task/partition/mode groups; record review scope/any unreviewed categories honestly. All protocol issues are saved in `protocol_issues_for_review.json` and remain in denominators. External surface flags produce `external_collision_rows_for_review.json` without altering questions or scores. High format violation/cap counts cannot be called evidence of absent knowledge. Preserve matches rather than regenerating examples to force zero.

## Local acceptance work and actual open issues

The unchanged candidate re-passes checksums/full validation with zero errors; all6,600 relational row answers/paths/source IDs were independently checked against canonical entities. The57-row static manual sample covers all nine task families and all twelve prompt partitions. Its wording/answers are coherent; arithmetic samples are correct. This structural review is not model final evaluation. Evidence is in `experiments/runs/data_qwen17_acceptance_review_20261106/`.

Pinned1.7B preliminary N token-budget audit passes:420 rows each,12,480 versus12,840 tokens (2.884615% within5%). This reuses the original bare continuation rendering, not a final v2/mode-specific supervised-token or optimizer/compute match; later continuation training still needs explicit token/masking/update budgets.

External review samples the first/middle/last entity per F/R/N_rel partition (nine units,27 separately quoted surface queries). It found real surface/context overlaps, so do not claim universal fictitiousness or collision-free surfaces. In particular, [Open Library catalogues The Silver Citadel under Anthony Horowitz](https://openlibrary.org/works/OL84773W/Silver_Citadel), while candidate R record `ku_000120` assigns that title to Brel Lume. [A primary university report names a Lustre Young Meteor prize](https://www.bcu.ac.uk/news-events/jewellery/celebrating-student-success); the fuller name/context differs from the fictional association. Website/artwork/game-name overlaps and inconclusive search-only hits are recorded separately, including failed page fetches.

These findings do not establish the candidate's invented author-book-prize graph as pre-existing, but the real-book title creates a referent ambiguity that requires an explicit collision-policy/version decision before full training. Keep candidate v1 and all prior evidence unchanged during this audit. If correcting a collision, use independent source evidence and a reproducible new candidate version, not a model-score-driven rename. Audit v1 can run while acceptance is pending; execution never silently accepts or edits it. Full-dataset acceptance, collision handling and scaled M1 protocol/budget remain open.

## Execution

Runner `src/evaluation/full_m0.py`; config `configs/evaluation/m0_qwen17_full_v1.json`; notebook `experiments/notebooks/10_m0_qwen17_full_audit.ipynb`. Publish new source/config/review evidence with authorization before the notebook's GitHub clone. Use existing Kaggle GPU kernel; optional HF_TOKEN is only for public-model downloads. No new weights/checkpoints are created or uploaded. Run sequentially and download the final evidence ZIP, then independently verify before any release/M1 scaling decision.

Local plan without model/tokenizer/network:

```bash
python3 -m src.evaluation.full_m0 --config configs/evaluation/m0_qwen17_full_v1.json \
  --output-dir /tmp/full-m0-plan-unique --mode plan
```

Actual inference uses `--mode audit` on Kaggle. Current local tests/preflight establish input/protocol logic only, not M0 model results.
