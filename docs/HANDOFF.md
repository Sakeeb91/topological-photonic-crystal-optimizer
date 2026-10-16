# Session Handoff

**Date:** 2026-10-16
**Branch:** `claude/dreamy-newton-ho6psr` (based on `main` @ `407df88`, the merge of
[PR #2](https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/pull/2)).
Earlier work is on `main` via [PR #1](https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/pull/1)
and PR #2.

## Status

- **Tests: 103 passed** locally (Python 3.13). CI (GitHub Actions) runs pytest on Python 3.11 and 3.13.
- **Published results regenerated** with the fixed geometry, seeded runs and real objectives:
  six single-objective runs (`results/run_20261016_*/`) and one full NSGA-III run
  (`results/multi_obj_20261016_015430/`). `docs/OPTIMIZATION_REPORT.md`, `docs/EXPLORATION_RESULTS.md`,
  `docs/RESEARCH_REPORT.md`, the README table and `docs/images/` are rewritten from them; every number
  was checked against the result files. The old research report's unsupported claims (67% improvement,
  MEEP validation, bending-loss law) are withdrawn in the new version.
- Both optimizers are seeded (`seed: 123` in every config) and replay exactly.
- **No real electromagnetic simulation exists.** All results come from two analytical stand-in models.
- Min-feature rule (user-approved): applies to the hole diameter, and to gaps a-2r, b-2r and (w-2r)/2.
- Repo-local git author is `Sakeeb91 <rahman.sakeeb@gmail.com>`. Environment does not persist; recreate the venv.

## Done this session (after PR #2)

| Commit | Change |
|---|---|
| `827b65a` | Runs record their config file; `compare_explorations.py` labels runs by it |
| `92a9964` | Multi-objective path seeded: NSGA-III, disorder model, simulation noise |
| `64c9f73` | Geometry "min feature size" reports the smallest printed dimension |
| `da930b5` | Six single-objective runs replaced with regenerated ones |
| `af7d977` | `docs/EXPLORATION_RESULTS.md` rewritten |
| `fa65e80` | Multi-objective run replaced with a full seeded run (pop 40, 50 gens) |
| `254033c` | `docs/OPTIMIZATION_REPORT.md` and `docs/images/` rewritten |
| `fd2fb4c`, `65c4e99` | README results table; `docs/index.md` image paths and captions |
| `2a0ef58` | `docs/RESEARCH_REPORT.md` rewritten; unsupported claims withdrawn |

## Known issues, in priority order

1. **The stand-in models contradict the project's own rules.** Changing them changes published numbers,
   so they were left as they are and documented:
   - `_simulate_physics_model` subtracts 10,000 when `r < 0.08 µm`. Every fabricable design now triggers
     it, so all single-objective scores carry a constant -10,000 offset.
   - `evaluate_design_mock`'s radiation term `1 - exp(-5/R)` falls as R grows, contrary to its comment
     ("Exponential improvement with R"), so the multi-objective run favors small rings.
   - The two models give Q on different scales; single- and multi-objective numbers are not comparable.
2. **MEEP integration** is still a placeholder: conda env with `pymeep`, a low-resolution test config,
   and a benchmark against the thesis's strong-dimerization design. This is what would make any result
   physical.
3. **Multi-objective disorder is applied twice.** `EnhancedDisorderModel` generates 8 disordered designs
   and each `evaluate_design_mock` call runs its own 8-run disorder loop, so q_std mixes two models.
4. **Installed package is named `src`.** Real CLIs need a package rename.
5. Minor: `src/analysis.py` "Iterations with improvement" counts increases over the previous iteration,
   not the best so far; `run_multi_objective_optimization.py` prints "Optimization completed" twice.

## Not started (low priority)

Dockerfile (useful for MEEP), pre-commit/black/flake8, CONTRIBUTING.md, `examples/` or notebooks,
`logging` instead of `print`, checkpoint/resume for long BO runs.

## Commands

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pytest
pytest                                                       # expect 103 passed
python run_optimization.py --config configs/strong_dimerization_v1.yaml   # 120 evals, ~2 min
python run_multi_objective_optimization.py --config configs/multi_objective_v1.yaml   # ~25 min
python src/analysis.py results/run_<TIMESTAMP>
python visualize_best_design.py results/run_<TIMESTAMP>
python compare_explorations.py
```

Branch rule for cloud sessions: develop and push on the session's `claude/…` branch, using
`git push -u origin <branch>`. Pushing to `main` returns 403; merge through a PR.
