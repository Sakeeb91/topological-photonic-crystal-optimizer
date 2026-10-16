# Parameter Space Exploration Results

Six Bayesian optimization runs, one per single-objective config, regenerated on 2026-10-16 at commit
`92a9964`. Each config sets `seed: 123`, so every run replays exactly.

**What these numbers are.** Every score comes from the analytical stand-in model
(`_simulate_physics_model` in `src/simulation_wrapper.py`), not from MEEP. The findings below describe
how that model behaves; they are not electromagnetic results. Three properties of the setup matter when
reading them:

- **Score** = mean Q − 2 × Q standard deviation over the config's disorder runs (`q_penalty_factor: 2.0`).
- **Fabricable holes only.** `a` and `b` are center-to-center hole spacings. A design counts as feasible
  when the gaps `a − 2r`, `b − 2r` and the edge clearance `(w − 2r)/2` all exceed the 0.05 µm minimum
  feature size. Infeasible designs are scored 0 without simulating.
- **Constant offset.** The model subtracts 10,000 from any design with `r < 0.08 µm`. Every design
  allowed by the current bounds has `r ≤ 0.07 µm`, so all scores carry the same −10,000 offset. Rankings
  are unaffected, but these scores cannot be compared with those published before 2026-10-16, which
  came from design spaces where neighboring holes overlapped.

## Results

| Exploration | Config | Evaluations | Infeasible | Best score | Found at | a (µm) | b (µm) | r (µm) | R (µm) | w (µm) | a/b | b − 2r (µm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extreme Dimerization | `explore_extreme_dimerization.yaml` | 23 | 7 | **23,344** | 19 | 0.600 | 0.130 | 0.039 | 18.00 | 0.507 | 4.62 | 0.052 |
| Large Rings | `explore_large_rings.yaml` | 23 | 3 | 14,987 | 15 | 0.450 | 0.175 | 0.048 | 15.00 | 0.700 | 2.57 | 0.078 |
| Strong Dimerization (baseline) | `strong_dimerization_v1.yaml` | 120 | 14 | 12,890 | 84 | 0.400 | 0.188 | 0.041 | 10.00 | 0.450 | 2.13 | 0.105 |
| Compact Designs | `explore_small_compact.yaml` | 23 | 9 | 11,335 | 20 | 0.346 | 0.133 | 0.030 | 6.00 | 0.361 | 2.61 | 0.073 |
| Short Test Run | `test_meep_v1.yaml` | 15 | 5 | 11,175 | 9 | 0.400 | 0.135 | 0.030 | 12.00 | 0.480 | 2.96 | 0.075 |
| Fabrication Limits | `explore_fabrication_limits.yaml` | 30 | 8 | 11,094 | 19 | 0.400 | 0.200 | 0.040 | 8.00 | 0.500 | 2.00 | 0.121 |

"Found at" is the 0-based evaluation index where the best score first appeared. Every best design is
feasible. Full logs, parameters and plots are in `results/run_20261016_*/`.

![Parameter Exploration Comparison](images/parameter_exploration_comparison.png)

## What drives the score in the stand-in model

1. **Dimerization dominates.** The model adds `15,000 × (a − b)/(a + b)`. The `extreme_dimerization`
   run, with a/b = 4.62, beats every other run by more than 8,000, and the best `a` sits at or within
   0.004 µm of its upper bound in all six runs.
2. **Ring radius goes to whichever bound the period favors.** The model gains 2,000 per µm of R but
   loses 100 per hole, and the hole count grows as R/(a + b). The net slope changes sign near
   a + b ≈ 0.63 µm: about −600 per µm at a + b = 0.48 (compact), roughly 0 at 0.63 (large rings),
   and +300 at 0.73 (extreme dimerization). That is why R lands on the lower bound in the
   strong-dimerization, compact and fabrication-limits runs and on the upper bound (18 µm) in the
   extreme-dimerization run.
3. **Hole radius and waveguide width barely matter.** Across the allowed range r = 0.03–0.07 µm the
   model's Q changes by about 400, well below the run-to-run disorder noise (the per-design Q standard
   deviation is roughly 10% of Q, about 1,000–2,000). Width enters only through a penalty that is
   already constant here. The best `r` and `w` values are therefore not informative.

These are properties of the stand-in model. Whether larger dimerization, ring radius or hole size
actually raise Q needs the MEEP simulation (see `docs/HANDOFF.md`).

## Reproduce

```bash
python run_optimization.py --config configs/strong_dimerization_v1.yaml   # and the five other configs above
python src/analysis.py results/run_<TIMESTAMP>
python compare_explorations.py                                           # writes parameter_exploration_comparison.png
```
