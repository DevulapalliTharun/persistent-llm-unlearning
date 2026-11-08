# Prospective M1 v2 — inverse prize-to-book support

Prepared2026-11-09. **No v2 GPU fit/training/evaluation/private upload has run.** Config `configs/training/m1_full_qwen17_inverse_v2.json`; runner `src/training/inverse_m1.py`; notebook15 `experiments/notebooks/15_m1_full_inverse_support.ipynb`.

## Hypothesis and preserved control

Verified v1 fails only thinking F prize-to-author21/30. The wording diagnostic improves that group to24/30 under familiar phrasing, but6 training-wording errors remain. Their visible traces choose another book/author; this motivates testing the inverse support edge, not a proven internal mechanism or guaranteed remedy. Existing training teaches book-to-prize and book-to-author directly, but not standalone prize-to-book.

Add direct prize-to-book supervision uniformly across all30 F+90 R units. Keep Qwen3-1.7B revision70d244cc86ccca08cf5af4e1e306ecf908b1ad5e, candidate v1.1 facts/splits/checksum, all seven original task targets/paraphrases/modes, LoRA/LR/loss/precision/batches/seed/max length, development inputs/decoding/gates and final-test isolation. Preserve the failed1200 model/metrics/receipts as frozen control. Source/control/report hashes are checked before preparation/execution. No final or later N question is model-evaluated/trained.

Start this comparison **fresh from the same public M0/new LoRA**, not from the trained v1 adapter or optimizer snapshot. This keeps starting weights consistent with v1 and avoids claiming unsupported exact mid-run resume. The old model remains privately backed up and unchanged.

## Training-only overlay

Bank `configs/data/m1_inverse_book_train_v1.json` has three fixed questions using only the prize placeholder, e.g. “Which book received Young Meteor Prize?” → “The Zephyr Nexus.” Templates are checked against reserved forms, source-placeholder restrictions, rendered collisions and answer leakage. All canonical answers/paths come from existing F/R records.

| Item | V1 control | V2 candidate |
|---|---:|---:|
| Knowledge units | F30/R90 | Same |
| Original target rows | 5040 | Exactly the same5040, unchanged prefix |
| Auxiliary rows | 0 | 720 =120 units×3 wordings×2 modes |
| Total rows | 5040 | 5760 |
| Unique questions | 2520 | 2880 |
| Rows by mode | 2520 each | 2880 each |
| Rows by F/R | 1260/3780 | 1440/4320 |
| Development | 840 per mode | Byte-identical original840 |

The auxiliary view is `m1_train_aux_inverse_book`, task `prize_to_book`, path prize ID→book ID, answer book.title. Thinking target supplies the same canonical book-received-prize sentence; non-thinking target is only Answer: title. This supporting book/prize edge is not relabelled as the designated author-book forget target. Auxiliary data is training-only, has no final-metric eligibility and creates no easier new acceptance test. No changes to generator TASKS, original manifest or source prompt files.

## Budget and selection declared before results

Keep microbatch4×accumulation4,rank8/alpha16/dropout0/all-linear,LR0.0003,masked per-example response CE,FP16 base/FP32 LoRA,clip1,seed42,max256.

Near-match nominal row exposure: ceil(1200×5760/5040)=**1372 attempted/required applied updates**. That is21952 examples and3.811111 nominal dataset passes versus v1's19200/3.809524; relative per-row difference0.0417%. This is an additive coverage intervention with approximately preserved original-row exposure, **not equal tokens/compute or an isolated causal attribution**. Extra updates, shorter auxiliary outputs and changed shuffle trajectories must be reported. Actual exposures remain logged.

Checkpoint/private critical-artifact round-trip at **343/686/1029/1372 attempts**; only endpoint1372 is eligible. No intermediate dev selection, automatic retry/resume, silent budget extension or mode/batch/precision fallback. Existing checkpoints preserve artifacts but do not implement exact loader/RNG mid-run resume.

## Reused execution and unchanged acceptance

Reuse the existing full-M1 training/scoring/gate/reload/private-backup functions without editing v1 source/config. Separate disposable memory probe must match this config/source/actual longest batch and exits with no saved adapter. Actual process loads fresh M0/new LoRA after original T4/package/tokenizer metadata checks;32 exact source fact controls+10 own-model utility outputs must reproduce before LoRA creation. Reuse1680 already-verified M0 dev records with provenance, preserving observed year matches and base protocol failures.

Endpoint/reload each evaluate840 dev questions in each mode+10 thinking arithmetic probes; same3422 new control/post/reload answers total. No auxiliary training-wording scores replace the declared dev gate. Full original family gates plus each original one-hop task>=90%/multi-hop>=75% per F/R/mode remain fixed. Require full840 coverage, measurable base utility>=80%,post drop<=10 points,zero post format/cap/boundary failures,all1372 updates,LoRA change,exact output/token reload and four private checkpoint receipts. The ten reused arithmetic probes remain narrow diagnostic evidence, not independent broad utility validation. Supplied/composite chains are supervised, not proof of unseen-composition reasoning.

`comparison_v1.json` preserves the original failed verdict/post metrics alongside the candidate's, with exposure/compute limitations. Canonical M1 remainsfalse until independent raw/private-persistence review. No M2 or safeguard claim follows merely from execution completion.

## Durability and local checks

Private snapshots/endpoint: `tharundevulapalli/persistent-llm-unlearning-checkpoints/shared/m1-full-qwen17-inverse-v2/<unique-run>`. Fresh subprocesses handle all model/service work; notebook parent caches do not supply packages/model state. `--mode backup` preserves the endpoint with hash round-trip or labels an incomplete evidence upload without claiming an endpoint. `--mode export` includes root text and12 checkpoint JSON metadata files, excludes adapter/optimizer binaries, and privately verifies the ZIP itself under `evidence/m1-full-inverse-v2/<unique-run>/`. Sidecar receipt records its immutable commit/SHA. Re-exporting unchanged evidence reuses the ZIP; changed evidence refuses overwrite. Download it too.

CPU plan/actual cached pinned-tokenizer preparation pass. All5760 prefixes/masks/EOS validate with no truncation; original5040 encoding/masking totals match frozen v1 exactly. Longest sequences remain107 answer-only/141 thinking,below256. New per-mode prompt counts255198/243678 and supervised-response counts20388/75405 are dataset totals, not runtime sample exposure. Evidence is `experiments/runs/m1_inverse_v2_preflight_20261109/`.

Local tests cover preserved old targets/dev/M0 baseline,canonical inverse labels/paths/modes,uniform F/R coverage,no N/reserved/label leakage,fixed recipe/budget/checkpoints/gates/private namespace,old gate reuse with new budget and failed21/30 retained,ZIP text/metadata selection/re-export refusal/private round-trip corruption. Actual GPU fit/scaled learning/real private uploads are untested. The stale temporary tokenizer environment was rebuilt with existing pinned requirements; no model weights/PyTorch were installed or loaded locally.

Publish new source/config/bank/tests/protocol/preflight with authorization before running notebook15's clone. Then execute prepare→disposable fit→fresh actual trial→endpoint/reload→private backup→ZIP export. Preserve failed outcomes/source comparison and independently verify results before canonical promotion or further recipe changes.
