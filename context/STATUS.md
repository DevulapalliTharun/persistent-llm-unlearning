# Current Project Status

**As of:** 2026-11-06

**Current phase:** 0.6B training-paraphrase run is verified but still fails thinking F composite. A controlled Qwen3-1.7B candidate comparison is prepared: same 504 rows/120 attempts/gates, own new M0 baseline and separate disposable GPU memory probe. Public metadata/tokenizer preparation and all 52 local tests pass; no 1.7B weights, GPU fit, baseline inference or learning result yet. Final model choice and canonical M1/M2 remain pending.

> This status distinguishes documented plans from inspected artifacts and logged experiments. Nothing is marked complete without saved evidence.

## What is already available

- [x] Two proposed project guides: `docs/proposals/RAFG.pdf` and `docs/proposals/RASP_Project_Guide.pdf` (visible in the provided folder screenshot).
- [x] Problem framed: persistence of targeted forgetting under later legitimate reasoning fine-tuning.
- [x] Shared pipeline agreed: M0 → M1 → M2 → undefended recovery pilot → separate RAFG/RASP branches.
- [x] Seven evaluation metrics identified; staged priorities agreed.
- [x] VS Code-centric, no-paid-API workflow discussed, with Colab/Kaggle as candidate compute backends.
- [x] Portable execution/storage architecture documented: GitHub for code and context, Kaggle as primary remote GPU, Colab as fallback, and private Hugging Face storage for durable model artifacts.
- [x] Private artifact repository created at `tharundevulapalli/persistent-llm-unlearning-checkpoints`; Kaggle Secrets authentication and a committed-version upload/download checksum round-trip were verified.
- [x] Reproducible M0 smoke-test runner and thin Kaggle/VS Code notebook implemented locally; syntax and notebook structure validated.
- [x] Interactive M0 smoke test passed through VS Code on a Kaggle T4 x2 runtime: CUDA, immutable model revision resolution, tokenizer/chat template, BF16 model loading and bounded inference were verified.
- [x] Fresh Kaggle Save & Run All version reproduced the pinned M0 smoke test from Git commit `bb50b38e2315626d1e2eae3dc719f85a5ead8561`.
- [x] Synthetic relational dataset schema and deterministic split/validation plan documented in `docs/data/DATASET_SPEC.md`; no accepted full pilot bundle generated yet.
- [x] Standard-library dataset generator, validators and reproducibility tests implemented; two independent tiny bundles matched byte-for-byte with zero validation errors.
- [x] Tiny Qwen tokenizer-budget audit passed: 636 `N_rel` vs 648 `N_reason` tokens (1.89% difference, within 5%); automatic near-duplicate checks found no pairs at similarity ≥ 0.92.
- [x] Tiny manual review passed: exact public searches found no collisions for the 9 sampled authors/books/prizes, prompt wording was coherent, and final evaluation was expanded to two exclusive paraphrases per task (126 tiny final prompts).
- [x] M0 infrastructure and smoke-test code verified on a fresh Kaggle committed run.
- [x] Dataset-generation and validation code implemented and verified in tiny mode.
- [x] Full 30/90/60 candidate saved at `data/candidates/pilot-v1/seed-42/`; two independent generations matched byte-for-byte and repository copy matched both.
- [x] Full automated validation passed with zero errors/near-duplicate pairs; tokenizer audit passed at 12,480 N_rel versus 12,840 N_reason tokens (2.88% difference). Nine unit tests pass.
- [x] M0 knowledge-audit runner, fixed config and VS Code/Kaggle notebook prepared; local preflight verifies 1,260 non-final prompts covering 180 units. All 13 local tests pass.
- [x] User-reported Kaggle M0 audit completed from Git commit `a774aa00b0b13ebdb748727abc741ecd86123401`: 1,260 predictions, 0 exact matches, 0 empty final responses, and 2 longer-response year mentions requiring review. Summary preserved in `experiments/runs/m0_knowledge_audit_20261105/reported_summary.json`.
- [x] Both pasted year flags reviewed against candidate records: one year embedded in an incorrect book/prize claim; one semantically correct year in prose. Preserve zero strict exact matches and record the prose answer separately in `experiments/runs/m0_knowledge_audit_20261105/match_review.json`.
- [x] All five original audit files downloaded and saved under `experiments/runs/m0_knowledge_audit_20261105/`; raw checksum and all 1,260 inputs/summary counts verified. A 42-response nonmatch sample across all 21 partition/task groups was reviewed; evidence is in `artifact_verification.json`.
- [x] Defined and implemented `answer-only-v2`: shared `Answer:` instruction/parser, 128-token cap, per-response EOS/cap logging, separate format/answer/mention counts and compatible future SFT target renderer. All 23 unit tests pass; pinned-tokenizer prompt rendering and 1,260-question CPU preflight pass.
- [x] User-reported v2 Kaggle audit completed from Git commit `b2df70757ed57b53270913ecd453d5fc36a7ac12`: 1,260 predictions, zero parsed-answer matches, zero format violations and zero token-limit hits. Concise summary saved in `experiments/runs/m0_answer_only_audit_20261105/reported_summary.json`.
- [x] All five original v2 files preserved; prediction/config/source/input hashes verified and all 1,260 scores/group counts recomputed. Raw distribution: 1,259 `Unknown` responses and one wrong book answer; all rows log EOS, generated counts 5–18. Inspected 42 samples across 21 groups plus the one exception. Evidence: `experiments/runs/m0_answer_only_audit_20261105/artifact_verification.json`.
- [x] Tiny M1 diagnostic code/config/notebook implemented for 4 F and 8 R units. All 29 local tests pass; actual pinned-tokenizer preparation verifies 84 training rows, 101-token maximum, 7,153 masked prompt tokens and 601 supervised answer/EOS tokens.
- [x] Selected 12-unit mappings/templates manually inspected and admitted only for the diagnostic. Fixed pilot learning, utility and save/reload gates documented in `docs/data/TINY_M1_PILOT.md`; full release/collision review remains pending.
- [x] Kaggle CUDA, pinned training-package imports, private artifact access and real 84-row preparation confirmed by user. Initial model loading succeeded.
- [x] Recorded first tiny M1 setup failure at adapter attachment: installed optional `torchao==0.10.0` rejected by PEFT (minimum 0.16.0). No optimizer updates or learning metrics. Evidence: `experiments/runs/m1_tiny_setup_failure_20261105/failure.json`.
- [x] User-reported retry overcame setup failure and completed all 120 attempts. Adapter changed, optimizer budget, format/cap and identical reload-prediction gates passed. F direct 8/8, F composite 11/12, R direct 16/16; R composite 16/24 failed the 75% threshold. Reported result saved in `experiments/runs/m1_tiny_attempt_20261105/reported_result.json`.
- [x] Recorded pilot failure honestly: base utility 0/10 fails the ≥8/10 measurable-baseline gate; post utility 4/10 does not establish meaningful preservation. Pilot remains failed; no threshold changes.
- [x] Preserved private HF round-trip receipt and all 14 original tiny M1 evidence files. ZIP/file hashes, input/config/source provenance, 282 scored predictions, 120 optimizer records and full post/reload equality verified locally; see `experiments/runs/m1_tiny_attempt_20261105/artifact_verification.json`. Adapter binaries remain private and were not re-downloaded locally.
- [x] Inspected all 11 failed development outputs and all utility responses. Base utility abstains on 10/10; post utility has 4 strict matches, 3 correct numbers with units and 3 wrong answers. R composite failures: 5 author-to-year (3 correct years with extra text) and 3 wrong prize-to-author responses. Original scores/gates unchanged; see `manual_review.json`.
- [x] Prepared versioned utility-prompt diagnostic: same pinned base/saved adapter, original and arithmetic-specific instructions, 40 inference-only answers, pinned private artifact hashes and original-response reproduction controls. All 31 local tests, notebook syntax and model-free plan passed before the now-completed GPU run.
- [x] Utility-prompt Kaggle run completed from `5d244de`; all eight original evidence files preserved under `experiments/runs/m1_utility_prompt_diagnostic_20261105/`. All 40 scores, source/config/prediction hashes, prompts, summary groups and original-response reproduction controls verified. Original-v2 base/adapter 0/10 and 4/10; arithmetic-v1 7/10 and 6/10; zero Unknown under arithmetic-v1.
- [x] Paired arithmetic review: base/adapter make the same three numerical errors; adapter additionally outputs `12 kilometers` instead of bare `12`. Manual numerical correctness is 7/10 for both. This is a ten-item instruction-sensitivity diagnostic, not general utility preservation; the original >=8/10 baseline and R composite gates remain unmet.
- [x] Prepared notebook 05 thinking-mode feasibility check: greedy reproduction plus matched sampled thinking/non-thinking, same ten utility questions on both checkpoints and all 84 development questions on the adapter (312 planned answers, zero training). Complete thinking boundaries required; visible reasoning saved separately from final-answer scoring. All 37 local tests passed before the now-completed GPU mode run.
- [x] Preserved/verified notebook 05 run from `3c77324`: all 14 original reports, all 312 predictions, 228 saved token records and three exact greedy controls pass local checks. Matched thinking arithmetic: base 10/10, adapter 10/10; base omits Answer: on three correct values. Matched adapter development: 73/84 non-thinking vs 21/84 thinking. See `experiments/runs/m1_thinking_mode_diagnostic_20261105/`.
- [x] Inspected all thinking development finals and selected visible traces. Paired changes: 53 lost matches, one gained, 20 correct in both, ten wrong in both. One thinking development response caps/uncloses; 62 other failed outputs complete. Invented/mixed associations and trace/final contradictions observed. No weight update or unlearning occurred; training/evaluation mode mismatch is a hypothesis, not an isolated causal finding.
- [x] Prepared mixed-mode tiny retry: same pinned base/12 records/original LoRA+optimizer budget, fresh adapter, 84 answer-only plus 84 canonical-chain thinking targets. Local pinned-tokenizer preparation verifies exact prefixes/masking, max lengths 101/134, 601/2,298 supervised response tokens and zero truncation. All 42 local tests pass. See `docs/data/TINY_M1_MIXED_PILOT.md` and `experiments/runs/m1_tiny_mixed_preflight_20261105/`.
- [x] Notebook 06 completed from `7d4b28f`: all twenty original reports preserved under `experiments/runs/m1_tiny_mixed_attempt_20261105/`. Config/source/data/target hashes, all 534 scores, 282 token records, 120 updates and full post/reload records (except batch time) verify. Private receipt records adapter round-trip at `25764276e3e83e6c20519429854e15216f57e676`.
- [x] Recorded mixed retry honestly: base fact scores 0/84 in both modes; post answer-only 78/84 with passing mode gates; thinking 63/84 with F direct 7/8 and R composite 16/24 failures plus one capped/unclosed response. Thinking utility remains 10/10. All post failures/traces reviewed; no gates changed or canonical M1 promotion.
- [x] Prepared notebook 07 wording comparison: 84 record/task pairs, two wordings/two modes plus ten utility controls (346 planned predictions, zero updates). Pins helper commit `7d4b28f` and private mixed adapter `25764276e3e83e6c20519429854e15216f57e676`; embeds config so VS Code can run it without publishing new helper files. All seven cells compiled and model-free pair/config/reference/source-hash checks passed before the now-completed GPU run.
- [x] Notebook 07 completed and all fourteen original files preserved under `experiments/runs/m1_training_wording_diagnostic_20261106/`. All 346 scores, 178 token records, three original raw controls and all 168 paired response/category records verify. Same private adapter/source pins; zero updates.
- [x] Wording diagnosis: training questions 80/84 in both modes; development 78/84 answer-only, 63/84 thinking. Thinking paired outcomes: 63 correct both, seventeen training-only, zero development-only, four wrong both. Answer-only: 77/3/1/3 respectively. Utility 10/10; residual exact-training errors are years/reverse lookup, so wording is not the only unresolved issue.
- [x] Prepared notebook 08 training-paraphrase retry: fourteen new source-only/exclusive templates, 252 unique questions/504 two-mode rows, same twelve records/120 attempts. Pinned-tokenizer prefix/masking/no-truncation preparation passes (max106 answer-only/max138 thinking). Original targets/development unchanged; 47 tests and previous-run verifiers pass. See `docs/data/TINY_M1_PARAPHRASE_PILOT.md` and `experiments/runs/m1_tiny_paraphrase_preflight_20261106/`.
- [x] Notebook 08 completed from `c6f53fe`; all 25 original files preserved under `experiments/runs/m1_tiny_paraphrase_attempt_20261106/`. All 870 scores, 450 token records, 504 targets/audit, 120 update records, verdict and post/reload equivalence verify. Private receipt records adapter round-trip at `f8fbea97a97cc50f3572c3efb62219dbf5820fa9`.
- [x] Recorded tradeoffs: thinking development 63/84→70/84 but F composite 10/12→6/12; R composite 16/24→19/24, direct both-mode author/book all correct. Original training probes 77/84 answer-only, 76/84 thinking; utility 9/10 and zero post format/cap/boundary failures. Only thinking F composite gate fails; no uniform-improvement or whole-pilot pass claim.
- [x] Prepared pinned Qwen3-1.7B comparison at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, with unchanged 504 targets/dev/gates and fresh own-model M0 utility/fact baseline. Public metadata/config/tokenizer fetched without model weights; tokenizer verifies all 504 prefixes/masks/no truncation. All 52 tests and notebook syntax/model-free plan pass. See `docs/data/QWEN17_CAPACITY_COMPARISON.md` and `experiments/runs/m1_tiny_qwen17_preflight_20261106/`.
- [ ] Run notebook 09 GPU fit first (temporary adapter/update discarded), then fresh 1.7B tiny trial and compare against preserved 0.6B control; verify/preserve all evidence before selecting a model.
- [ ] Full synthetic candidate accepted after manual collision review and M0 accidental-knowledge audit.
- [ ] M1 and M2 checkpoints produced.
- [ ] Recovery demonstrated and measured.
- [ ] RAFG or RASP method results collected.

## Stage gates

| Stage | Current status | Exit evidence |
|---|---|---|
| 0. Repository + environment | **Complete:** Git baseline, private artifact round-trip, pinned environment, interactive GPU test and committed M0 rerun verified | Git baseline, environment/config file, remote GPU smoke test |
| 1. Synthetic data | Full automated gates, original/v2 audit verification and response sampling passed; remaining manual collision/acceptance review and pilot criteria precede training | `experiments/runs/m0_answer_only_audit_20261105/`, protocol, candidate data and original evidence |
| 2. M0 → M1 SFT | Tiny paraphrase run trained/reloaded/persisted; answer-only passes but thinking F composite 6/12 fails; overall pilot still failed | `experiments/runs/m1_tiny_paraphrase_attempt_20261106/` plus preserved earlier runs |
| 3. M1 → M2 unlearning | Not started / unverified | Saved canonical M2, reduced target accuracy, reasonable retain/reasoning accuracy |
| 4. Ordinary recovery pilot | Not started / unverified | From M2, logs across continuation steps for unrelated and similar-structure conditions |
| 5A. RAFG trials | Future | Trigger logging, recovery/utility/overhead vs fixed and periodic guards |
| 5B. RASP trials | Future | Module scores, recovery/utility/overhead vs uniform, random and forget-only controls |
| 6. Final comparison | Future | Same M2, matched tests/cost, seed variability, limitations |

## Immediate next actions (in order)

1. After user-authorized publication of the Qwen3-1.7B comparison/public metadata/control evidence, run `experiments/notebooks/09_m1_tiny_qwen17_comparison.ipynb` on Kaggle/HF_TOKEN. First measure representative GPU fit in a disposable process; if successful, start a fresh own-model M0 baseline and the unchanged 120-attempt training recipe. No 1.7B GPU run has occurred yet. Further 0.6B-only budget tuning is deferred while suitability is compared.
2. Use preserved failure/task breakdown and declared sampling limits to plan further learning experiments. Do not infer capability solely from response CE, pick a favourable sample, lower thresholds or exclude failed records. Keep final-test isolation; constructed-chain training does not independently demonstrate unseen composition, and the repeatedly used arithmetic probes are development checks.
3. Select the next model only after comparing learning/utility/fit evidence. A selected new model needs its own full candidate knowledge/token-budget/release checks before scaling beyond the admitted 12-unit diagnostic; old 0.6B results/adapters remain separate.
4. Scale to a first reproducible **M1**, then implement and validate **M2**.
5. Only after M2 is validated, run **ordinary post-unlearning recovery pilot**. Branch into RAFG/RASP after the pilot.

## Known risks and checks

- **M1 fails to learn:** check question/answer formatting, chat template, label masking, learning rate, and train/held-out results.
- **M2 fails to forget or loses utility:** check Gradient Difference sign and coefficient balance, retain sampling and checkpoint lineage.
- **No recovery after later SFT:** this is a possible research outcome; test both preplanned continuation conditions and report honestly.
- **Weak small-model reasoning:** use controlled one- to three-hop tasks as primary reasoning evaluation; harder datasets are optional.
- **Cloud notebook resets:** persist adapters, data manifests and results outside ephemeral runtimes at every milestone.
- **Office laptop handoff:** perform a clean restore test on the personal laptop before January 29; comply with employer device policy.

## Reporting rules

Do not write percentages, plots, speed claims or method wins into `STATUS.md` until they are produced by logged experiments. When runs exist, link each claim to its config, checkpoint and results file.
