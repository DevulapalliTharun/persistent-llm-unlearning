# Verified M0 knowledge-audit artifacts — 2026-11-05 IST

The user ran the audit from VS Code on a Kaggle Tesla T4 and downloaded the original output ZIP. All five original files are preserved here, and all 1,260 prediction rows have been independently rescored. Their results exactly reproduce `summary.json`. The earlier `reported_summary.json` transcription and `reported_matches_for_review.json` also agree with the originals. [`artifact_verification.json`](artifact_verification.json) records the checks and a 42-response review sample.

- Source Git commit, from the supplied preflight: `a774aa00b0b13ebdb748727abc741ecd86123401`.
- Pinned model revision: `c1899de289a04d12100db370d81485cdf75e47ca`.
- Coverage: 1,260 non-final prompts across 180 candidate knowledge units.
- Counts by entity partition: F=210, R=630, N_rel=420.
- Normalized exact matches: 0, including 0 exact year matches.
- Answer-mention review flags: 2, both in `author_to_year`, one R and one N_rel.
- Empty final responses: 0.
- Runtime: FP16 on `cuda:0`, Tesla T4; reported duration 117.4098 seconds; reported peak allocated GPU memory 1.2503 GiB.
- UTC run: 2026-11-04 20:16:21 to 20:18:19; local IST date: 2026-11-05.

Observed interpretation: the pinned model produced no whole-string exact expected answers under this audit's wording, decoding, precision and matching rules. This does not prove the model has no relevant knowledge, or that it gave no semantically correct answers.

## Review of the two year mentions

The user supplied both flagged rows, preserved in `reported_matches_for_review.json`. The review against local canonical records is in `match_review.json`.

1. **R / Elira Sable / 2019:** The candidate graph is *The Wandering Delta* → *Young Elm Prize* → 2019. The generated response names a different book (*The Last Days of the World*) and the Booker Prize. The expected year appears, but the supporting associations disagree with the candidate.
2. **N_rel / Zella Pryce / 2023:** The response states the requested year correctly in a full sentence. This is a semantically correct year answer despite failing whole-string exact matching. The response alone cannot distinguish chance from prior knowledge of the synthetic association.

Keep the original metric (zero whole-string exact matches) unchanged and report the manual finding separately. Do not describe this as zero correct answers. Define response-format handling before comparing M0/M1/M2 and retain all predictions. No candidate records have been changed or excluded based on these rows.

## Preserved and verified artifacts

Remote directory: `/kaggle/working/m0-knowledge-audit-feenm09g`.

`summary.json`, `predictions.jsonl`, `matches_for_review.json`, `plan.json` and `environment.freeze.txt` were extracted unchanged from `m0-knowledge-audit-feenm09g-results.zip`. The independently verified predictions digest is:

```text
c5bc23c7ce65bf86b46deca5f84f90bcbf90146177c59577cea5ca2612fa80b8
```

Verification checked the ZIP integrity and five expected entries, prediction checksum, 1,260 unique example IDs, all original input fields against the candidate, config/input-file hashes, and every summary group/count by recomputing the original scoring. The two original flagged rows agree with the pasted records and existing manual review. The local files also match the ZIP byte-for-byte.

## Nonmatch review and evaluation limitations

Inspected 42 responses: the first and middle nonflagged row from each of the 21 partition/task groups. The sample contains invented book/author/prize claims, unrelated answers and refusals. Several responses visibly end mid-sentence under the 32-token generation cap. The original run did not log EOS-versus-cap termination or generated-token counts, so an exact truncation rate is unavailable.

The baseline is valid as a record of this particular audit. Before M1/M2 comparisons, define consistent answer extraction, an adequate output cap and termination logging. If a revised audit is run, keep this original baseline and report the protocol change. Do not assume zero whole-string matches means zero knowledge or zero semantically correct answers. Full manual collision review and dataset acceptance remain pending; no dataset regeneration or training has occurred.
