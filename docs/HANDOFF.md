# Session Handoff

**Date:** 2026-10-10
**Branch:** `claude/dreamy-newton-ho6psr`, open as
[PR #2](https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/pull/2) into `main`.
Earlier work (tests, seeds, config validation) was merged to `main` in
[PR #1](https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/pull/1) (`f0ed67b`).

## Status

- **Tests: 95 passed** locally (Python 3.13). **CI** (GitHub Actions, `.github/workflows/tests.yml`)
  runs pytest on Python 3.11 and 3.13 for every PR and push to `main`.
- Both optimizers run end to end. `run_optimization.py` scores designs with overlapping holes as
  infeasible (0.0) without simulating them. `run_multi_objective_optimization.py` runs with both
  multi-objective configs, including `advanced_multi_fidelity_v1.yaml` (previously had no feasible design).
- **Published results are stale.** `docs/OPTIMIZATION_REPORT.md` and `docs/EXPLORATION_RESULTS.md`
  carry a note: their designs come from design spaces with overlapping holes, and the multi-objective
  metrics are proxies. See issue 1.
- **No real electromagnetic simulation exists.** `evaluate_design_meep` still returns the analytical
  `_simulate_physics_model` Q-factor.
- Repo-local git author is `Sakeeb91 <rahman.sakeeb@gmail.com>`. Environment does not persist; recreate the venv.

## Done on this branch (PR #2)

| Commit | Change |
|---|---|
| `d96d135` | Design vector built by name; saved `run_config.yaml` files (sorted keys) previously scrambled parameters on re-run |
| `9b7b254` | `optimization_log.csv` stores the real score (was negated, so analysis picked the worst design) |
| `1ff9aa9` | Sign corrected in the 6 tracked historical logs (verified against `best_params.yaml`) |
| `cb1b8d8`, `f75715d` | Analysis reports, plots and `docs/images/` figures regenerated; one report sentence corrected |
| `d9d6add` | GitHub Actions CI |
| `8c23f1f` | `get_min_feature_size`: reads `constraints.`, then `fabrication.`, then top-level `min_feature_size` |
| `4676914` | Multi-objective runs receive full objectives (`simulation.return_comprehensive_objectives` was ignored) |
| `2855f65` | `--output-dir` creates `plots/` and `designs/` |
| `cf506ed` | Fabrication constraints declared to pymoo (`n_ieq_constr=2`); clean exit when nothing is feasible |
| `d48b37e` | `hole_clearance_violations` (a-2r, b-2r, (w-2r)/2 > min feature) in both single-objective validators; min-feature check on hole **diameter** |
| `ea6216c` | Config bounds: r shrunk to ~[0.03, 0.07] so holes fit (user-approved table); `advanced_multi_fidelity_v1` gains `objective.num_disorder_runs` |
| `f0f06d2` | Infeasible designs scored 0.0 without simulating; validators reject boxes with no feasible point |
| `48b0f6b` | Notes on both reports that results predate these fixes |

Decisions this session: `a`, `b` are center-to-center spacings; keep a, b near the ~0.45 µm Bragg period
and shrink holes; infeasible corners penalized, not excluded.

## Known issues, in priority order

1. **Regenerate published results** with the fixed geometry and objectives: the 6 single-objective
   exploration runs (configs `strong_dimerization_v1`, `explore_*`, `test_meep_v1`) and one
   multi-objective run, then update both reports and `docs/images/`. Needs the user's go-ahead since it
   replaces published numbers. Full runs take ~10-30 min each with the mock.
2. **The mock physics rewards large holes** (`optimal_r = 0.15` in `_simulate_physics_model`, `0.12` in
   `evaluate_design_mock`), so optimizers push r to the feasibility boundary. Its conclusions about hole
   size are meaningless for the new bounds; MEEP (issue 6) is the real fix.
3. **Seed doesn't reach the multi-objective path.** NSGA-III, `src/multi_objective_optimizer.py` and
   `src/active_learning.py` use global `np.random`.
4. **Multi-objective disorder is applied twice.** `EnhancedDisorderModel` generates N disordered designs
   and each `evaluate_design_mock` call runs its own `num_disorder_runs` loop: N x M mock evaluations
   per design, and q_std mixes two disorder models.
5. **Installed package is named `src`.** Real CLIs need a package rename.
6. **MEEP integration** is still a placeholder: conda env with `pymeep`, a low-resolution test config, and
   a benchmark against the thesis's strong-dimerization design.
7. Minor: `src/analysis.py` "Iterations with improvement" counts increases over the previous iteration,
   not over the best so far.

## Not started (low priority)

Dockerfile (useful for MEEP), pre-commit/black/flake8, CONTRIBUTING.md, `examples/` or notebooks,
`logging` instead of `print`, checkpoint/resume for long BO runs.

## Commands

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pytest
pytest                                                       # expect 95 passed
python run_optimization.py --config configs/strong_dimerization_v1.yaml   # 120 evals, mock physics
python run_multi_objective_optimization.py --generations 4               # quick NSGA-III check
python src/analysis.py results/run_<TIMESTAMP>
python visualize_best_design.py results/run_<TIMESTAMP>
```

Branch rule for cloud sessions: develop and push on the session's `claude/…` branch, using
`git push -u origin <branch>`. Pushing to `main` returns 403; merge through a PR.
