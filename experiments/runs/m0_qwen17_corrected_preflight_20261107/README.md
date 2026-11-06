# Corrected M0 refresh preflight — 2026-11-07

`plan.json` is a model-free plan, not a GPU result. It verifies the checksum-pinned v1/v1.1 candidates, frozen reviews and completed/independently verified parent evidence. Three changed development rows invalidate the complete eight-row batch at offset832. Two unchanged16-row control batches cover both views. Running both modes therefore plans64 controls plus16 affected-batch answers, then a derived2520-row baseline with2504 unchanged parent records if every control field matches.

Source Git HEAD is the existing published base; new working files are additionally identified by exact source hashes. CPU preparation does not imply these new files are published. No model/adapter/weights, GPU generation, corrected baseline, dataset acceptance or full M1 training occurred here. See [protocol](../../../docs/data/PILOT_V1_1.md).
