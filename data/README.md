# Dataset bundles

The full seed-42 pilot candidate is in [`candidates/pilot-v1/seed-42/`](candidates/pilot-v1/seed-42/).

It contains 30 forget, 90 retain and 60 new relational knowledge units, the prompt files, split manifest, validation report and checksums. These small synthetic text files belong in Git so every agent can use the same dataset.

**Candidate status:** automated generation, checksum, reproducibility, near-duplicate and tokenizer-volume checks passed on 2026-11-05. Full manual collision review and the base-model accidental-knowledge audit remain pending. Training has not started.

Validation evidence is in [`experiments/runs/data_full_validation_20261105/`](../experiments/runs/data_full_validation_20261105/).

Verify the saved candidate:

```bash
python3 -m src.data.validate data/candidates/pilot-v1/seed-42
```

The generator refuses to overwrite a nonempty directory. To reproduce this candidate, generate into a new temporary directory using the same config and generator files. `manifest.json` records the baseline Git commit, exact generator-source hashes and config hash; the source hashes identify the correction made after that baseline commit. Once those changes are committed, keep the manifest's original provenance when comparing the saved bundle. A later Git commit changes manifest provenance even if the generated facts are identical.

`manifest_created_at_utc` is a fixed value from the versioned config so generation remains reproducible; the actual audit date is recorded in the experiment summary and work log.

Keep the candidate unchanged while auditing it. If a correction is necessary, preserve the original evidence and record a new candidate version. The accepted bundle will move to `data/generated/` after the remaining dataset gates pass.
