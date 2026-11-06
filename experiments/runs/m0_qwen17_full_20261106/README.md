# Verified full Qwen3-1.7B M0 audit — 2026-11-06

Notebook 10 completed on Kaggle from source commit `667a2b836091be7a4c3c7cc4c5a6ae3d9a64b3b0`. This is fresh-base evaluation, with **no adapter, training or weight updates**. The passing tiny M1 adapter remains a separate experiment; canonical full M1/M2 do not exist yet.

Model: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, FP16 on one Tesla T4. Unchanged candidate v1 contains 30 F, 90 R and 60 N_rel units. Each mode evaluates 840 F/R development questions and 420 N_rel training-view questions, independently batched with the declared seeds. No final-test inference or N_reason evaluation occurred.

| Result | Answer-only | Thinking |
|---|---:|---:|
| Questions | 1,260 | 1,260 |
| Parsed final-answer matches | 9 | 1 |
| Non-year matches | 0 | 0 |
| Format violations | 809 | 752 |
| Token-limit hits | 0 | 16 |
| Missing thinking boundaries | 0 | 15 |
| Expected-answer mentions in visible reasoning | 0 | 4 |

All final matches and reasoning mentions are year values. The nine answer-only matches are bare years without the requested label; the frozen parser accepts them and records format violations. Frequent default year outputs and invented supporting entities make these results compatible with coincidental matches, but their cause is not established. They remain correct under the declared primary score. No non-year author/book/prize answer matches were observed. These observations do not certify absent internal knowledge, especially given the format/cap problems.

All 13 potential-answer rows were reviewed, including all four flagged thinking traces. One additional thinking final says `Answer: 2000, Booker Prize`: the year is correct, but the extra prize is incorrect, so the frozen primary score remains false. The reasoning flags do not demonstrate the canonical synthetic chains and are not post-unlearning leakage measurements. See `manual_review.json` for every priority interpretation, 78 stratified protocol samples, all 16 capped-response tails/metadata and the 112 collision-flagged final fields. The complete 1,562 protocol-issue traces were not all manually read; every row was mechanically verified.

The independently established real-book title collision remains unresolved: candidate `ku_000120` assigns *The Silver Citadel* to Brel Lume. Both model modes instead answer John Gardner for book-to-author; this neither matches the synthetic author nor resolves the catalogue's different attribution. See the frozen [external review](../data_qwen17_acceptance_review_20261106/README.md). **Dataset accepted=false**. Next resolve the confirmed title ambiguity in a reproducible new candidate version, preserving v1 and these scores; revalidate affected data/baselines before full M1. No correction or new training was implemented in this review.

The completed archive `m0-qwen17-full-55wpqjfh-evidence.zip` has SHA-256 `2cd4e6e534bb4c0d61fec334e8ec7a42755e6e9564111c6e724aaaab99beb75b`. Its nine original text files are preserved byte-for-byte here. `artifact_verification.json` records their hashes and checks. The review files intentionally duplicate selected raw responses; the largest text files are about 10 MB and contain no model checkpoints. Runtime was 4,073.7561 seconds (about 68 minutes), peak allocated memory 5.275 GiB. Package versions and generation metadata are preserved.

Verification reproduces all 2,520 input/order/score/flag/token records, both full grouped summaries, review selections and frozen config/source/input provenance. Optional offline replay additionally checks every prompt and decoded output using the exact cached tokenizer, without loading weights or running inference:

```bash
python3 experiments/runs/m0_qwen17_full_20261106/verify_artifacts.py
# Optional, using an environment with Transformers and the pinned tokenizer cached:
python3 experiments/runs/m0_qwen17_full_20261106/verify_artifacts.py \
  --tokenizer-cache /path/to/cache
```

Both checks passed locally; the tokenizer replay used a temporary cache outside the repository. All 57 unit tests, notebook syntax/error-output checks, archive byte comparison, credential-pattern scan and Git whitespace checks also pass. Executed notebook outputs are preserved. No new model artifact, commit or push was created during this review.
