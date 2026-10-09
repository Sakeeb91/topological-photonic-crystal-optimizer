# Session Handoff

**Date:** 2026-10-09
**Branch:** `claude/dreamy-newton-ho6psr` (pushed to origin)
**Base:** `main` @ `e4e66b6`. This branch merges in the earlier review branch
`claude/review-code-011CUSEGWPp4DhiXCfDCMaNi`, so it carries all work from both
sessions. **None of it is on `main` yet.** The user chose to open/merge the PR
themselves.

## Status

- **Tests: 68 passed, 0 failed** (`pytest`).
- `run_optimization.py` runs end-to-end, and two runs of the same seeded config
  produce byte-identical `optimization_log.csv` files.
- `run_multi_objective_optimization.py` runs end-to-end with
  `configs/multi_objective_v1.yaml` (`--generations 4`, exit 0).
- **No real electromagnetic simulation exists.** `evaluate_design_meep` still
  returns the analytical `_simulate_physics_model` Q-factor; all MEEP code is
  commented out.
- **Environment does not persist.** Recreate the venv every session.
- Repo-local git author is `Sakeeb91 <rahman.sakeeb@gmail.com>` (set by request
  this session; earlier commits keep their Claude author, history not rewritten).

## Done this session (previous known issues 1-7)

| Commit | Change |
|---|---|
| `b46f39c` | Merge the previous review branch (tests, setup.py, packaging) with main's repo reorganization |
| `1a42c46` | `create_parameter_summary` skips derived quantities for vectors shorter than 5 |
| `0eb8a22` | Untrack `src/__pycache__/*.pyc` (committed before `.gitignore` existed) |
| `6e0b474` | Remove broken `console_scripts` entry points from `setup.py` |
| `a5c4305` | Simulation functions take an optional `rng` (`np.random.Generator`); default is seeded from `config['seed']` |
| `d58af9f` | Top-level `seed` config key drives `gp_minimize` and a per-run Generator; validated; `seed: 123` in single-objective configs |
| `7752922` | `run_multi_objective_optimization.py` uses `from src....` imports (no `sys.path` hack) |
| `da9b412` | `validate_multi_objective_config`, including a check that the design space has a feasible point |

Decisions: `results/` and `*.png` stay gitignored (add curated figures with
`git add -f`).

## Known issues, in priority order

1. **Logged scores are negated.** `run_optimization.py` writes `-score` to
   `optimization_log.csv` (the comment says the opposite). `src/analysis.py` and
   `compare_explorations.py` take `max()` as "best", so they report the **worst**
   design. Tracked historical logs under `results/run_*` have the same sign
   error; `best_params.yaml` is correct. Check whether `docs/OPTIMIZATION_REPORT.md`
   and `docs/EXPLORATION_RESULTS.md` used the wrong values. Ask before rewriting
   committed result data.
2. **`configs/advanced_multi_fidelity_v1.yaml` cannot produce any feasible
   design** (`b_max - 2*r_min = 0.04 <= min_feature_size 0.05`), and its
   `objective` section lacks `num_disorder_runs`. The validator now rejects it,
   so the command in `README.md` exits with that message. Fixing it means
   choosing new physical bounds: user's call.
3. **Multi-objective config keys that are silently ignored.**
   `constraints.min_feature_size` is never read (the optimizer reads a
   top-level `min_feature_size`, default 0.05), and
   `simulation.return_comprehensive_objectives` is never read (the mock checks
   the top level), so the optimizer falls back to approximating Q as
   `score + 20000` with no real bandgap/mode-volume values.
4. **`multi_objective_v1.yaml` feasible region is ~3% of the box**, so
   generation 1 often has zero feasible designs; with `--generations 1` the run
   crashes with a NaN traceback in `generate_design_recommendations`.
5. **`--output-dir` doesn't create `plots/` and `designs/`**, so the multi-objective
   run crashes when saving the first plot unless those exist.
6. **Seed doesn't reach the multi-objective path.** `src/multi_objective_optimizer.py`
   and `src/active_learning.py` still use global `np.random`, and NSGA-III is not
   seeded. If `seed` is set in a multi-objective config, every `evaluate_design_mock`
   call reuses the same draws (fresh Generator per call).
7. **Installed package is named `src`.** Real CLIs need a package rename.
8. **MEEP integration** is still a placeholder. Largest piece of real work left;
   needs a conda env with `pymeep` and a fast low-resolution test config.

## Not started (low priority)

CI (GitHub Actions running pytest), Dockerfile (useful for MEEP),
pre-commit/black/flake8, CONTRIBUTING.md, `examples/` or notebooks, `logging`
instead of `print`, checkpoint/resume for long BO runs.

## Commands

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pytest
pytest                                                       # expect 68 passed
python run_optimization.py --config configs/strong_dimerization_v1.yaml   # 120 evals, mock physics
python run_multi_objective_optimization.py --generations 4               # quick NSGA-III check
python src/analysis.py results/run_<TIMESTAMP>
python visualize_best_design.py results/run_<TIMESTAMP>
```

Branch rule for cloud sessions: develop and push on the session's `claude/…`
branch, using `git push -u origin <branch>`. Pushing to `main` returns 403.
