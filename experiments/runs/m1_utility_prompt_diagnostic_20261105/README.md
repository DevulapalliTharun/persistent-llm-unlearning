# Verified utility prompt diagnostic — 2026-11-05

All notebook 04 code cells completed on Kaggle from Git commit `5d244def80cd70b0407df143322d092b1a2b48e7`. This was inference only: forty answers, zero training updates, no requested/scored reasoning traces. Original tiny M1 pilot remains failed; no canonical M1/M2 was produced.

| Checkpoint | Original-v2 strict correct | Original-v2 Unknown | Arithmetic-v1 strict correct | Arithmetic-v1 Unknown |
|---|---:|---:|---:|---:|
| Base | 0/10 | 10/10 | 7/10 | 0/10 |
| Saved tiny adapter | 4/10 | 0/10 | 6/10 | 0/10 |

Both original-v2 response sets reproduce the original training run exactly. All four groups have zero label-format violations and token-limit hits. Runtime: Tesla T4, FP16, torch 2.11.0+cu130, Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0; reported evaluation time 4.2309 seconds excludes artifact/tokenizer downloads and includes base loading/evaluation/adapter attachment.

## What the responses show

Changing the full system instruction eliminated the base model's abstentions on these ten questions. The model can answer seven correctly under the alternative instruction; the original zero score is not proof of absent arithmetic ability. This comparison changes the complete instruction, so it does not isolate the Unknown clause alone.

Under arithmetic-v1, both models make the same three numerical errors: two boxes of five pens answered 20 rather than 10; double six minus five answered 12 rather than 7; twenty chairs minus eight plus three answered 12 rather than 15. The adapter's additional strict mismatch is `Answer: 12 kilometers` versus the expected `12`; the base returns `Answer: 12`. Manual numerical correctness is therefore 7/10 for both. This does not establish broad utility preservation or degradation, and the strict results stay 7/10 and 6/10.

The base's alternative score still falls below the original >=8/10 measurable-baseline gate. No gate is lowered and no original result is replaced. The separate R composite development failure also remains unresolved.

## Evidence and checks

Remote output: `/kaggle/working/m1-utility-prompt-9ypiagyk`. Archive `m1-utility-prompt-9ypiagyk-evidence.zip`, SHA-256 `40f8efa30e59f90ed63382f2ece882b47a5cbc5215dc7f69dbdbd669d17b84df`. All eight original JSON/JSONL/text files are preserved unchanged. ZIP CRC, original bytes/file hashes, all prediction/config/source hashes, forty input/score records, group summaries, prompt strings, token-count/end-marker consistency, original-output reproduction controls and environment versions pass local verification. See `artifact_verification.json` and `manual_review.json`.

The private source-artifact receipt records a fresh download/hash verification of the same adapter at immutable revision `07d23d8b904ab48401f9c6676d05bb85a14d9fbd`. Adapter binaries and restored original reports were excluded from the small ZIP. Local checks compare the receipt to the already-preserved original evidence/adapter manifest; no binary re-download or model/tokenizer replay occurred locally. EOS identity was not independently loaded from the model config; saved token lengths/final-token consistency was checked against observed token 151645. The plan's `model_executed=false` is a pre-execution snapshot; the completed summary records inference.

Recheck from any directory:

```bash
python3 /path/to/repo/experiments/runs/m1_utility_prompt_diagnostic_20261105/verify_artifacts.py
```

Next: define the next utility protocol/probe prospectively and address the preserved R composite errors before another M1 attempt. Preserve strict/manual findings separately, keep evaluation arithmetic out of training, and leave final-test data untouched. No new model, threshold, training setting or diagnostic protocol was selected during this review.
