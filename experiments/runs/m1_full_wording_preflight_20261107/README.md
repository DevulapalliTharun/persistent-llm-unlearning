# Full-M1 wording diagnostic preflight — 2026-11-07

Model-free plan and pinned-tokenizer layout only; no diagnostic GPU output or new training. `plan.json` declares1722 predictions:42 exact source controls, then840-row overlay in both modes. Only120 original-training-wording prize-to-author questions replace dev rows, with720 unchanged dev fillers, original full batch positions/seeds and30 F/90 R target pairs per mode. `selected_overlay.jsonl` preserves this order; original source data/results/gates remain unchanged.

`prompt_layout.json` renders3360 original/overlay prompts across two modes with the exact cached1.7B tokenizer. Exactly120 prompts change per mode,720 remain identical; no maximum padded-prompt-length changes in any53 original batches per mode. No model weights/PyTorch/GPU loaded locally. This is not evidence of diagnostic accuracy or a causal mechanism.

Notebook14 embeds the tested config/helpers and uses published35b961e inference/scoring code, so new source publication is unnecessary for runtime execution. All86 local code tests pass, including embedded/source synchronization. Real adapter restore, replay controls,1722 inferences and private ZIP backup remain untested pending Kaggle execution. See `docs/data/FULL_M1_WORDING_DIAGNOSTIC.md`.
