# Candidate v1.1 CPU review — 2026-11-07

The dedicated correction command produces two byte-identical copies of the separate candidate v1.1. All checksums/structural checks pass. Only R ku120's canonical title changes, from The Silver Citadel to The Veylith Citadel, plus entity-version metadata. Splits/IDs and all other facts are preserved; selected tiny12 facts/train/dev inputs are unchanged. All6600 relational rows follow exactly the declared substitution, with no model final-test use. N_rel/N_reason prompts are byte-identical to v1. See `verification.json` and the [correction/refresh protocol](../../../docs/data/PILOT_V1_1.md).

Offline pinned1.7B tokenizer preparation renders2520 prompts across both modes and rechecks the preliminary legacy N token budget in `token_budget.json`. This is not final supervised-token/compute matching. No model weights or GPU inference were used.

External replacement review is bounded and not proof of universal uniqueness. Original v1 and all audit/tiny evidence are preserved. Notebook11's64 replay controls and16 affected-batch answers are planned only. Corrected GPU results, manual output review, dataset acceptance, full M1 protocol/training and M2 remain pending. No commit/push performed during this preparation.
