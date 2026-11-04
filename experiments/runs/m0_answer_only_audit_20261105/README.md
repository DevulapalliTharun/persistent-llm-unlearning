# Verified answer-only M0 audit — 2026-11-05 IST

The user executed `02_m0_answer_only_audit.ipynb` on Kaggle through VS Code and downloaded the original ZIP. All five original files are preserved here. Independent rescoring reproduces the full summary, and `reported_summary.json` agrees with its supplied fields. `artifact_verification.json` records checksum, input/source verification and response inspection.

## Verified evidence

- Protocol: `answer-only-v2`, pinned Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Preflight source Git commit: `b2df70757ed57b53270913ecd453d5fc36a7ac12`.
- Preflight config digest: `6c871ad523fae774de9dd9c27883ce33086955cea4c7362f710b886ec575597e`.
- Preflight runner digest: `8cd7af093900dbd67acca5f5226d39f826d8181ea512828ca8d3838b60a36ad9`.
- Preflight answer-protocol digest: `b80a1315b0fc0c0d29909ca4e2e671dd81036dd96f47a328cb95e7c587c362fe`.
- 1,260 completed predictions over all 180 candidate units using non-final prompts.
- Parsed-answer matches: 0; year matches: 0; non-year matches: 0.
- Format violations: 0. Token-limit hits: 0.
- FP16 on Tesla T4; reported elapsed time 38.4952 seconds and peak allocated GPU memory 1.325 GiB.

The original predictions reveal **1,259 `Answer: Unknown.` responses and one incorrect book response**. The sole exception is `ex_m1_dev_ku_000017_author_to_book_01`: asked for Cira Farrow's book, the model supplied *The Last Days of the World* rather than the candidate's *The Echoing Zenith*. Thus the zero score primarily reflects abstention under the fixed instruction. It does not establish absent internal knowledge or successful unlearning.

All 1,260 rows log an EOS stop, with generated-token counts from 5 to 18 and zero empty final responses. Recomputed format/answer counts agree with the summary; termination metadata is internally consistent. Raw token IDs were not saved, so EOS/cap logging cannot be independently replayed token by token. `format_compliant` measures the label/single-line structure and does not certify that the field contains only a concise entity.

Manually inspected 42 first/middle responses across all 21 partition/task groups (all `Unknown`) plus the single non-Unknown exception. The original v1 baseline remains separately preserved; it used different prompting and should not be compared as an improvement or evidence of learning.

## Preserved artifact files

Remote output directory: `/kaggle/working/m0-knowledge-audit-v2-n1fck5ni`.

`plan.json`, `summary.json`, `predictions.jsonl`, `matches_for_review.json` and `environment.freeze.txt` were extracted without changes from `m0-knowledge-audit-v2-n1fck5ni-results.zip`. ZIP integrity and its exact five entries were checked. All 1,260 unique example IDs/input fields match the canonical candidate, all config/input/runner/parser hashes agree, original review flags reproduce, and the full v2 answer-group summary was independently recomputed. The reported concise-summary fields also match.

Verified predictions SHA-256:

```text
3bd2761acf1e45a4396fac5b9e5a0a86d01d01df1b1c5301d0d4ca7e08180f92
```

Run timestamps are November 4 UTC / November 5 IST. Original artifacts are now stored outside Kaggle and can be committed to Git. This run is accepted as reproducible evidence of M0 behaviour under v2. Full manual collision review, final dataset acceptance, M1 learning thresholds and training remain pending. M1 must learn direct and multi-hop targets with the same protocol before low M2 accuracy can be interpreted as forgetting.
