# Training-wording diagnostic v1

Status: all seven notebook 07 cells completed and all 346 predictions independently verified on 2026-11-06 IST. Same saved mixed adapter, zero updates. See `experiments/runs/m1_training_wording_diagnostic_20261106/`.

The mixed retry has low training response CE but 78/84 non-thinking and 63/84 thinking development accuracy. Measure generation on the actual training questions before choosing another SFT change. Low teacher-forced loss does not establish generation correctness, and training-question accuracy is not held-out performance.

## Fixed comparison

Use all 84 original training questions, each paired with the same record/task/answer in the original 84 development questions. Collapse the two training-format copies of a question into one question for each evaluated mode. Align both views to the original development ordering, preserving identical batch positions. Require different question wording, disjoint template IDs and unchanged canonical answers; do not select only failures.

| Suite | Questions | Generation |
|---|---:|---|
| Training wording, non-thinking | 84 | Original greedy, 128-token cap |
| Development wording, non-thinking | 84 | Same greedy settings |
| Training wording, thinking | 84 | Temperature 0.6, top-p 0.95, top-k 20, min-p 0, 1,024-token cap |
| Development wording, thinking | 84 | Same sampled settings |
| Utility reproduction, thinking | 10 | Same sampled arithmetic settings |

Total 346 answers, zero optimizer updates. Batch size 16, seed 42 plus batch offset, same instruction within each fact-mode comparison. Compare wording within each mode; comparing thinking to greedy also changes sampling/cap. This is one sampled seed, not a generalization estimate across seeds. Preserve all format/cap/boundary flags and raw visible reasoning/finals. A target only in unfinished thinking does not count, using the existing scorer unchanged.

The adapter is restored from private HF revision `25764276e3e83e6c20519429854e15216f57e676`, path `diagnostics/m1-tiny-mixed-v1/m1-tiny-mixed-v1-_avmkde0`, on the same pinned base. Hash-check the adapter and six original input/summary/reference reports. Reproduce the original 84+84 development and ten utility raw responses before generating training-wording predictions. On a mismatch, save the controls and stop before interpreting wording effects.

Save all selected questions, plan/config snapshot, helper-source hashes, artifact verification, generation metadata, five prediction files, stage metrics, all paired final responses/categories, environment and control comparisons. No model upload or new adapter is needed. The final evidence ZIP excludes restored binary files. Release the inference model from GPU memory after evaluation.

## Interpretation fixed before execution

- Strong training-wording generation with weaker development wording supports a wording/generalization limitation in this setup, rather than showing that more optimizer steps alone are necessary.
- Weak performance on both views leaves training-question generation/learning reliability unresolved; inspect traces, final answer types and caps before attributing a cause.
- Thinking-specific gaps warrant examining mode conditioning as well as prompt variation. One sampled draw cannot establish a general mode mechanism.

Do not treat training-question scores as final accuracy, select a favourable sample, lower thresholds or promote the original mixed pilot. These are diagnosis directions, not conclusions already observed. No final-test/N prompts, further SFT or unlearning are used.

## Running without another source-publication dependency

Open `experiments/notebooks/07_m1_training_wording_diagnostic.ipynb` locally in VS Code and use the existing Kaggle GPU kernel/HF_TOKEN. The notebook pins the already-published helper commit `7d4b28f1d18d543ea0daacf8f09cb2df9dbf0eed` and embeds its small config snapshot; the separate review copy is `configs/evaluation/training_wording_diagnostic_v1.json`. It therefore does not require the new notebook/config or the latest mixed-run reports to be present in GitHub before execution: VS Code sends the notebook cells to the remote kernel, existing evaluation code comes from the pinned clone, and immutable evidence comes from private HF.

The first cell clears cached imports only for the project's `src` package so earlier temporary clones do not determine helper versions. Follow cells sequentially from the top, always creating fresh output/restore directories. New notebook orchestration is kept in the notebook; reusable scoring, inference, data selection and metadata helpers are the existing published implementations. This is a small orchestration exception to the earlier CLI-runner pattern, avoiding a publication blocker without changing evaluators. The plan saves the complete embedded config, intended suites and hashes of pinned source files for reproducibility. Git commit/push still requires user authorization when publication is requested; this workflow does not push anything automatically.

After the run, download the evidence ZIP and verify reproduction and paired counts before choosing further training changes. The completed run restored/verified the private artifact and reproduced all three original raw-output controls; no local model/tokenizer replay occurred during independent review.

## Verified outcome — 2026-11-06

Both modes answer 80/84 exact training questions. Development is 78/84 answer-only and 63/84 thinking; utility remains 10/10. Thinking paired counts are 63 correct both, seventeen training-only correct, zero development-only correct and four wrong both. Answer-only paired counts are 77/3/1/3. Training-question groups have no format/cap/boundary failures; development thinking retains the original one capped/unclosed response. All 346 scores, 178 token records and 168 paired response/category records verify.

These observations support wording-sensitive generation under this setup, especially thinking, while four errors on exact training wording in each mode show incomplete generation reliability too. Do not treat this one-seed comparison as proof that wording is the sole cause, perfect fact memorization, or a new pilot pass. Next proposed training correction is a versioned mixed-mode tiny retry with additional new training-only paraphrases and declared exposure budget, keeping development/final prompts and all gates intact. No such augmentation or new SFT was implemented/executed in review.
