# Candidate acceptance preparation — 2026-11-06

The original seed42 candidate is unchanged. Current checksum/invariant validation passes with zero errors; all6,600 relational row answers/paths/source IDs match canonical entities. The existing57-row static sample was reviewed and covers all task families/prompt partitions, including final templates only for structural review. See `structural_review.json`.

The cached pinned1.7B tokenizer gives preliminary continuation token totals12,480 N_rel versus12,840 N_reason across420 rows each, a2.884615% difference within5%. Detailed counts/provenance are in `token_budget.json`. This is the legacy bare rendering, not final supervised-token/compute matching.

Bounded external collision review is in `collision_review.json`. Initial OR searches were inconclusive;27 separately quoted surface queries across nine preselected units found context overlaps. Confirmed real-book title collision: *The Silver Citadel* is catalogued under Anthony Horowitz, while candidate `ku_000120` assigns it to Brel Lume. University reporting also uses the fuller phrase Lustre Young Meteor prize; this is not proof of the fictional book/prize link. Search-only website/artwork/game/character hits and failed fetches are labelled, not treated as verified book/author associations. Negative/no-exact-snippet searches are inconclusive and do not certify absence.

**Dataset is not accepted or promoted.** Full1.7B M0 audit, explicit book-title collision handling and scaled-M1 protocol/budget remain pending. No names, expected answers, splits or final templates changed; no model was evaluated on final-test questions. Independent external evidence may justify a reproducible candidate version correction, but no automatic filtering/regeneration based on M0 correctness is allowed.

Supporting direct sources: [book catalogue record](https://openlibrary.org/works/OL84773W/Silver_Citadel), [university award report](https://www.bcu.ac.uk/news-events/jewellery/celebrating-student-success). The audit runner/config pin these review snapshots by hash so their scope and unresolved issues remain traceable.
