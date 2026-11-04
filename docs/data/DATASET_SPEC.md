# Synthetic Relational Dataset Specification

**Status:** Generator implemented and tiny-tested; accepted pilot data not generated yet
**Version:** `v0.1-design`  
**Pilot seed:** `42`  
**Purpose:** Controlled M0 → M1 → M2 and post-unlearning recovery experiments

## 1. Scope

The pilot dataset teaches a model fictional author–book–prize–year relationships, removes a designated subset, and tests whether those relationships recover during later legitimate fine-tuning.

This is a controlled synthetic-fact study. It does not represent real personal data, certify deletion from model weights, or by itself establish general behavior across all LLMs.

No record or prompt described here has been generated or tested yet. All sizes remain configurable; the pilot defaults must pass validation before training.

## 2. Terminology and partitions

A **knowledge unit** is one connected fictional relationship graph:

```text
author ──wrote──> book ──won──> prize ──awarded_in──> year
```

The designated forget target is the **author–book association**, not the entire graph. Prize and year edges remain supporting facts so direct suppression can be tested through multi-hop paths. Direct forget tasks ask author ↔ book; multi-hop forget tasks traverse author/book through prize and year. Non-target supporting questions for an `F` unit are diagnostics and are not counted as forget or recovery successes.

The pilot contains 120 M1 knowledge units:

| Partition | Pilot count | Role |
|---|---:|---|
| `F` — forget | 30 | Learned at M1, targeted during M1 → M2 unlearning, evaluated for suppression and recovery |
| `R` — retain | 90 | Learned at M1, protected during unlearning and later fine-tuning |
| `N_rel` — new relational | 60 provisional | Disjoint new knowledge used only after M2 to test useful learning and structurally similar benign relearning |
| `N_reason` — unrelated reasoning | Budget-matched to `N_rel` examples | Math/logic tasks used only after M2 as a structurally different continuation condition |

`T` is **not an entity partition**. It is a held-out final-test prompt view generated from `F`, `R`, and where applicable `N`. `T` must never be used for training, monitoring, threshold tuning, module scoring, checkpoint selection, or early stopping.

The `N_rel=60` default is provisional. It may change after generator and tiny-SFT validation, but never in response to final-test performance.

## 3. Canonical knowledge-unit schema

Each unit is stored independently from generated prompts:

```json
{
  "schema_version": "1.0",
  "dataset_version": "pilot-v1",
  "record_id": "ku_000001",
  "partition": "F",
  "author": {
    "id": "author_000001",
    "name": "Mira Venn"
  },
  "book": {
    "id": "book_000001",
    "title": "The Glass Observatory"
  },
  "prize": {
    "id": "prize_000001",
    "name": "Silver Comet Prize"
  },
  "award_year": 1987,
  "forget_target": {
    "relation": "author_book",
    "direct_task_types": ["author_to_book", "book_to_author"],
    "multi_hop_task_types": ["author_to_prize", "author_to_year", "prize_to_author"]
  },
  "generation": {
    "seed": 42,
    "generator_version": "unimplemented",
    "source": "synthetic"
  }
}
```

### Required invariants

- `record_id`, author ID, book ID and prize ID are globally unique.
- Normalized author names, book titles and prize names are globally unique across `F`, `R`, and `N_rel`.
- Every author maps to exactly one book in the pilot graph.
- Every book maps to exactly one author and one prize.
- Every prize maps to exactly one book and one award year.
- Years may repeat, so a year is never used alone as a reverse-lookup key.
- Names and titles are synthetic and must not intentionally refer to real people, books or prizes.
- Automatic normalized-surface comparison rejects author, book and prize pairs with similarity at or above `0.92`; real-world collision review remains manual.
- Partition assignment is stored in a separate split manifest and copied into records only as a convenience; the manifest is authoritative.

## 4. Deterministic entity-first splitting

Entities are generated first, assigned to partitions second, and rendered into prompts only after the split is frozen.

### Pilot procedure

1. Generate 120 candidate M1 knowledge units using dataset seed `42`.
2. Validate global IDs, normalized surface forms and graph uniqueness.
3. Compute a stable split key for each record:

   ```text
   SHA256("pilot-v1|42|" + record_id)
   ```

4. Sort by the hexadecimal split key.
5. Assign the first 30 records to `F` and the remaining 90 to `R`.
6. Generate `N_rel` from a separate namespace and derived seed; it must not reuse an M1 entity or surface form.
7. Freeze and checksum the entity and split manifests before prompt generation.

The split must not depend on Python hash randomization, library shuffle behavior, model scores, training loss, or final evaluation results.

For later multi-seed experiments, each seed produces an independent manifest. Final seed values and count will be fixed before final comparisons; at least three training seeds are planned.

## 5. Prompt-example schema

Prompts are derived artifacts linked to canonical units:

```json
{
  "schema_version": "1.0",
  "example_id": "ex_final_f_2hop_000001",
  "record_id": "ku_000001",
  "entity_partition": "F",
  "prompt_partition": "final_test",
  "task_type": "prize_to_author",
  "hop_count": 2,
  "template_id": "final_2hop_prize_author_01",
  "question": "Who wrote the book that received the Silver Comet Prize?",
  "answer": "Mira Venn",
  "target_field": "author.name",
  "source_entity_ids": ["prize_000001"],
  "path": ["prize_000001", "book_000001", "author_000001"],
  "eligible_metrics": ["multi_hop_forget_accuracy", "recovery_rate"]
}
```

The training renderer may later convert `question` and `answer` into chat messages, but the canonical prompt record remains model-agnostic. Tokenization and answer-only label masking are training concerns, not embedded in the dataset schema.

## 6. Supported task families

### Direct / one-edge

- Author → book: “Which book did `<author>` write?”
- Book → author: “Who wrote `<book>`?”
- Book → prize: “Which prize did `<book>` receive?”
- Prize → year: “In which year was `<prize>` awarded?”

### Two-hop

- Author → book → prize
- Prize → book → author
- Book → prize → year

### Three-hop

- Author → book → prize → year
- Year is an answer or terminal attribute, never an ambiguous starting key by itself.

### Paraphrases

Each task family has disjoint wording banks for training, development, monitoring/scoring, guard updates and final evaluation. A paraphrase is not considered held out if it differs only by punctuation, capitalization or trivial word substitution.

The initial implementation contains 11 natural-language variants per task family and assigns exactly one exclusive variant to each prompt partition. Whether additional within-partition variants are needed remains a pilot-review question.

## 7. Prompt partitions and permitted use

| Prompt partition | Entities | Permitted use | Forbidden use |
|---|---|---|---|
| `m1_train` | `F + R` | M0 → M1 supervised fact learning | Final reporting |
| `m1_dev` | `F + R` | M1 checkpoint selection and learning diagnosis | Final reporting |
| `unlearn_f_train` | `F` | Gradient Difference forget batches on direct author ↔ book targets | Supporting prize/year erasure or final evaluation |
| `unlearn_r_train` | `R` | Gradient Difference retain batches | Final evaluation |
| `m2_dev` | `F + R` | M2 checkpoint selection using predefined criteria | Final reporting |
| `rasp_score_f` | `F` | Direct author ↔ book forget-gradient module statistics | RAFG triggering or final evaluation |
| `rasp_score_r` | `R` | Matched direct author ↔ book retain-gradient statistics | RAFG triggering or final evaluation |
| `rafg_monitor` | `F` | No-gradient direct and multi-hop target monitoring and trigger calculation | Gradient updates or final evaluation |
| `rafg_guard_train` | `F` | Direct author ↔ book corrective updates after a trigger | Trigger calibration or final evaluation |
| `n_rel_train` | `N_rel` | Post-M2 related-structure continuation | Contains no `F`/`R` entities or answers |
| `n_reason_train` | Independent tasks | Post-M2 unrelated-reasoning continuation | Contains no synthetic M1 entity strings |
| `final_test` (`T`) | `F + R`, and `N` where applicable | Final direct, paraphrased, 2-hop and 3-hop metrics | Any training, tuning, monitoring, scoring or selection |

Template IDs are exclusive to one prompt partition. The generator must fail if the same template ID or normalized template appears in incompatible partitions.

## 8. Preventing leakage

The validator must reject a dataset when any of the following occurs:

1. A record or entity ID occurs in more than one entity partition.
2. A normalized author, book or prize surface form occurs in more than one knowledge unit.
3. An `F` surface form appears in `R`, `N_rel`, or `N_reason` training text.
4. An `R` surface form appears in `N` training text.
5. A final-test answer appears verbatim inside its question.
6. A final template or normalized equivalent occurs in a training/development/monitoring template bank.
7. Two prompt records have identical normalized question text.
8. A graph query has zero answers or more than one valid answer.
9. A reverse lookup starts from a non-unique value such as year alone.
10. A later-task example contains an exact forgotten answer, target name or target title.
11. Train and final prompt files share an `example_id`.
12. Any generated name is blank, malformed, reserved, or outside configured length limits.

Additional audits should flag suspicious token overlap, substrings and near-duplicates for manual review rather than silently deleting them.

## 9. Recovery eligibility

Recovery is not computed over all `F` records automatically. A forget target is eligible only if it:

1. Was answered correctly under the predefined M1 rule.
2. Met the predefined forgetting rule at M2.
3. Has an unchanged final-test item across all continuation methods.

```text
Recovery@t = recovered eligible F targets at step t
             --------------------------------------
             total eligible F targets
```

The eligibility denominator, per-record M1/M2 status and exact matching rule must be saved before comparing no-protection, RAFG and RASP runs.

## 10. New-task conditions

Two post-M2 continuation conditions are preplanned:

### `N_reason`: unrelated reasoning

- Small generated arithmetic/logic or controlled reasoning tasks.
- No author, book, prize or exact M1 answer strings.
- Used to test whether general useful learning can incidentally recover forgotten facts.

### `N_rel`: similar relational structure

- New fictional author–book–prize–year units with a disjoint namespace.
- Same schema and comparable prompt style, but no M1 facts.
- Used to test whether structurally similar benign learning creates stronger recovery.

The two conditions should use comparable optimizer/update budgets. Example counts may be budget-matched after tokenization; raw row counts alone are not sufficient.

The pinned audit uses the Qwen3-0.6B tokenizer at revision `c1899de289a04d12100db370d81485cdf75e47ca`, non-thinking chat formatting and a maximum 5% total-token difference while preserving equal row/update counts. The tiny implementation test measured 636 `N_rel` versus 648 `N_reason` tokens (1.89% difference) and passed. The full pilot requires its own audit.

## 11. Planned generated layout

The generator will eventually produce a small, Git-trackable dataset bundle:

```text
data/generated/pilot-v1/seed-42/
├── manifest.json
├── checksums.sha256
├── entities/
│   ├── all_m1.jsonl
│   ├── forget.jsonl
│   ├── retain.jsonl
│   ├── new_relational.jsonl
│   └── split_manifest.json
├── prompts/
│   ├── m1_train.jsonl
│   ├── m1_dev.jsonl
│   ├── unlearn_f_train.jsonl
│   ├── unlearn_r_train.jsonl
│   ├── m2_dev.jsonl
│   ├── rasp_score_f.jsonl
│   ├── rasp_score_r.jsonl
│   ├── rafg_monitor.jsonl
│   ├── rafg_guard_train.jsonl
│   ├── n_rel_train.jsonl
│   ├── n_reason_train.jsonl
│   └── final_test.jsonl
└── reports/
    ├── validation.json
    └── manual_review_sample.json
```

No accepted pilot bundle exists yet. The implementation has generated only temporary tiny bundles for tests. File names may be refined during implementation, but partition meanings and leakage boundaries must not be weakened silently.

## 12. Manifest requirements

`manifest.json` must record:

- Schema, dataset and generator versions.
- Git commit and generation command.
- Master seed and all derived seeds.
- Requested and actual counts by entity/prompt partition and task type.
- Template-bank version and IDs.
- Normalization rules.
- Validation results.
- SHA-256 digest of every payload file. `checksums.sha256` additionally covers the manifest and all payloads and excludes only itself to avoid a circular digest.
- Timestamp in UTC.
- Known limitations and any manual exclusions with reasons.

## 13. Pre-training acceptance gate

Dataset generation is accepted only when:

- Counts match the frozen configuration (`F=30`, `R=90`, and reviewed `N` counts).
- Every required invariant and leakage check passes with zero errors.
- All direct, two-hop and three-hop paths resolve to exactly one answer.
- Final template IDs and normalized forms are absent from all non-final partitions.
- The manifest and checksums reproduce identically from a clean run with the same seed and Git commit.
- A manual sample covers every task type and prompt partition.
- M0 is audited against final synthetic facts to identify accidental exact-name knowledge; collisions are regenerated before M1.
- No final-test prompt has been used to tune generator rules based on model performance.

Passing this gate authorizes only a tiny M1 learning pilot, not full training.

## 14. Decisions still pending before implementation

- Final `N_rel` unit count; `N_reason` keeps equal rows/steps and must pass the 5% tokenizer-budget gate.
- Whether to expand beyond the implemented one exclusive template per task and prompt partition.
- Manual policy for accidental real-person, real-book and real-prize collisions.
- Exact M1 learning, M2 forgetting and recovery decision thresholds.
- Final multi-seed values beyond the pilot seed.

These must be resolved from design constraints or development data—not final-test outcomes—and recorded in `context/DECISIONS.md`.
