# Archived — pre-seed-fix episode data

This directory holds episode data generated before the seed-independence fix
(episode_runner.py bug: env.reset(seed=seed) reused one seed across all 30 episodes
in a block instead of 100*seed + episode_idx). Superseded by results/data_seedfix/.

Archived 2026-07-14. See findings.md (Phase 9.5) and CLAUDE.md for the full writeup
of the bug and its impact.
