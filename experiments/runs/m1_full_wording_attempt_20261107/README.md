# Verified full-M1 wording diagnostic — 2026-11-07

All six notebook14 cells completed. The downloaded ZIP is preserved byte-for-byte and all1722 new responses/240 target pairs independently verify, including pinned offline tokenizer replay. **Zero training updates. Original full M1 remains failed on its development gate.**

| Prize-to-author | Saved dev wording | Original training wording (diagnostic only) |
|---|---:|---:|
| F, non-thinking | 27/30 | 27/30 |
| R, non-thinking | 79/90 | 81/90 |
| F, thinking | 21/30 | 24/30 |
| R, thinking | 69/90 | 73/90 |

All42 exact fact/utility controls reproduce. All1440 unchanged filler records match every saved field/token, not just final answers; original batch positions/seeds are retained. Source/overlay target outputs have zero format/cap/boundary failures. Thus these target score changes are substantive output changes under the tested question forms, not parser/truncation failures. One seed and visible-text limits remain; cross-mode comparison also changes greedy versus sampled decoding/budget and is not a pure causal mode test.

Thinking F pairs:20 correct both,4 training-only,1 dev-only,5 wrong both. The original nine failures improve on ku1/32/46/53; ku45 newly fails under training wording. Six training-wording F answers remain incorrect. All six generated traces select a noncanonical book for the queried prize and then give another learned book's author. Across all23 wrong thinking training-wording target answers,22 omit the canonical book. This is observable trace mixing, not proof of an internal causal mechanism or absent knowledge.

All49 problem-pair finals were reviewed; both traces for34 thinking problem pairs (68 trace views) were read, including all10 F problem pairs and24 R cases. Non-thinking has15 problem pairs. All240 pairs are mechanically verified; correct thinking traces were not all manually read. `manual_review.json` preserves source/training outputs and canonical mappings. Wording affects some answers, while errors persist with familiar wording; development wording alone does not explain all observed failures. The earlier interpretation of model intelligence/causal undertraining is not established by this diagnostic.

Archive `m1-full-wording-ns6_8984-evidence.zip`,199349 bytes, SHA256 `9f60c5d5e1b1c640801aeb6554d0a79852b9764e9ccfb6523ee0a22cf17d18cf`, passes CRC/exact14 entries. All14 original files remain unchanged. `reported_result.json`/`reported_zip_persistence.json` are historical saved-output records; `local_verification.json` is the independently verified current record.

The same ZIP was privately uploaded and hash-round-tripped at revision `13e64fe99ac06396bcdf8978e45aa618e55cf8e2`, path `diagnostics/m1-full-wording-v1/m1-full-wording-ns6_8984/m1-full-wording-ns6_8984-evidence.zip`, repo `tharundevulapalli/persistent-llm-unlearning-checkpoints`. Its reported SHA matches the actual downloaded archive. Original model endpoint remains `a98bc31bcb3f7c404ea91ead35689301ab0c40d3`, helper source35b961e. No local model-weight download, private server re-query or new inference was performed; adapter binary restore relies on runtime hash records.

Verification reconstructs config/helper/script hashes (executed diagnostic_source.py matches the issued notebook),840-row overlay,120 training-wording substitutions, all1722 inputs/prompts/seeds/token IDs/decoding/scores/flags,240 paired entries/group metrics,42 controls,1440 exact fillers,source nine evidence/three adapter hash records and versions. All original full-M1 metrics/gates stay unchanged. Runtime287.244 seconds (~4.8 minutes),T4,original pinned packages.

Run `python3 experiments/runs/m1_full_wording_attempt_20261107/verify_artifacts.py`; add `--tokenizer-cache /path/to/cache` for actual offline prompt/token replay. Both forms pass. This verifies evidence, not canonical M1 acceptance.

**Next:** prospectively choose a bounded training improvement for reverse-association reliability/generalization across all F/R, then re-evaluate the original dev checks. The explicit prize→book→author direction merits attention because wrong generated traces often bind the prize to a different book. Exact intervention/budget has not been selected here. Preserve the failed1200 model and unchanged criteria; no new recipe, dataset/gate/mode change, checkpoint selection, training, canonical promotion, M2, commit or push in this review.
