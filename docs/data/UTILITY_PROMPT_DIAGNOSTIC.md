# Utility prompt diagnostic v1

Status: implemented locally; no diagnostic GPU run or new model result yet.

The tiny M1 run's base model answered `Unknown` to all ten arithmetic probes. Its trained adapter gave four exact answers, three correct numbers with units and three wrong answers. A zero starting score cannot establish useful preservation. Before changing training, test sensitivity to the evaluation instruction on the same checkpoints and questions.

The separate notebook is `experiments/notebooks/04_m1_utility_prompt_diagnostic.ipynb`; reusable code is `src/evaluation/utility_diagnostic.py`, with fixed config `configs/evaluation/utility_prompt_diagnostic_v1.json`.

## Fixed comparison

| Checkpoint | Original answer-only-v2 instruction | Arithmetic-specific instruction |
|---|---|---|
| Pinned Qwen base | Ten original utility questions | Same ten questions |
| Saved tiny adapter on pinned base | Same ten questions | Same ten questions |

Only the system instruction changes between prompt variants. The alternative says: `Solve the arithmetic question. Return exactly one line: Answer: <integer>. Give only the integer after the label, without units or explanation.` It changes the complete wrapper, so it cannot isolate the effect of the original `Unknown` clause alone. Both variants use non-thinking greedy decoding, 128 new tokens, left padding, original batch size 16 and seed 42. There are forty predictions, zero training updates and no traces requested or scored.

The fixed model revision is `c1899de289a04d12100db370d81485cdf75e47ca`. The tiny adapter is restored from the private repository at immutable revision `07d23d8b904ab48401f9c6676d05bb85a14d9fbd`, path `diagnostics/m1-tiny/m1-tiny-run-8myx_22r`. The downloaded original summary/reference predictions must match the config's preserved SHA-256 values; adapter files must match the summary's manifest. No live Kaggle training directory is required.

## Outputs and interpretation

Save four prediction JSONL files, raw responses, rendered prompts, generated token IDs, EOS/cap diagnostics, strict v2 scores, abstention counts, reference-reproduction comparisons, source/config/adapter hashes, versions and environment freeze. Original output strings must reproduce for both checkpoints under original-v2 before a new difference is attributed to the prompt. If they do not, preserve the output and diagnose environment/loading differences first.

The scorer is unchanged. Unit-bearing correct numbers remain strict mismatches and can be reviewed manually in the exported rows. Do not treat this diagnostic as a new training pass, retroactively change the failed tiny pilot, select the best prompt by final-test performance, or claim general reasoning preservation from ten questions. No F/R final-test or N training examples are used.

If the alternative yields a usable base score, it supports sensitivity to the full instruction in this setup; define any future utility protocol separately before another training run. If it does not, inspect raw errors before deciding whether another model, probe or setting is warranted. Neither outcome has been observed yet.

## Execution

The notebook clones GitHub `main` like the earlier notebooks. Commit/push the new diagnostic files before running it remotely; it stops immediately if the clone lacks them. No commit or push is performed automatically. Attach `HF_TOKEN` in Kaggle Secrets, select the Kaggle CUDA kernel, and run cells sequentially. The notebook restores the private adapter, performs inference, prints all answers and exports a small evidence ZIP. It does not upload new weights or reports automatically.

Local plan without GPU, model loading or network:

```bash
python3 -m src.evaluation.utility_diagnostic \
  --config configs/evaluation/utility_prompt_diagnostic_v1.json \
  --output-dir /tmp/utility-prompt-plan-unique --mode plan
```

Run GPU inference with `--mode evaluate` through the notebook. Use a fresh output directory each time. Download its evidence ZIP for independent review before another experiment.
