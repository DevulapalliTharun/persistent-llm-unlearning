
# Agent Instructions

This repository contains an M.Tech research project on persistent unlearning in reasoning-capable LLMs.

## At the start of every task
1. Read context/PROJECT.md.
2. Read context/STATUS.md.
3. Read context/DECISIONS.md.
4. Read recent entries in context/WORK_LOG.md.
5. Inspect relevant existing code before making changes.
6. Follow the active Git branch's research direction.

## During work
- Reuse existing implementations where possible.
- Keep experiments reproducible.
- Distinguish planned work from tested results.
- Never invent experimental results.
- Preserve common baselines for fair comparisons.

## After completing significant work
1. Update context/STATUS.md.
2. Append a concise entry to context/WORK_LOG.md.
3. Update context/DECISIONS.md if a decision changes.
4. Record files changed, tests run, results, blockers and next steps.
5. Report any incomplete work clearly.

## Git rules
- main contains the shared baseline up to M2 and recovery testing.
- rafg contains RAFG-specific work.
- rasp contains RASP-specific work.
- Do not merge branches or push changes without authorization.
- Do not commit API keys, credentials, or large model checkpoints.
