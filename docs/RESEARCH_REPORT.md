# Machine Learning Optimization of Topological Photonic Crystal Ring Resonators

## Research Summary

Regenerated 2026-10-16 from seeded runs at commit `92a9964`. Detailed results:
[EXPLORATION_RESULTS.md](EXPLORATION_RESULTS.md) (six Bayesian optimization runs) and
[OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) (NSGA-III multi-objective run and recommended designs).

---

## Status of the evidence

**Every result here comes from analytical stand-in models, not electromagnetic simulation.** The MEEP
FDTD path in `src/simulation_wrapper.py` exists only as commented-out code; `evaluate_design_meep`
returns the analytical `_simulate_physics_model` Q. The findings therefore describe how the stand-in
models respond to the design parameters. They do not yet establish anything about real devices.

An earlier version of this report (June 2025) claimed a 67% Q-factor improvement, a MEEP validation
campaign, an empirical bending-loss law (Q ∝ R^2.3) and several efficiency metrics. None of these were
supported by the code or data: no MEEP simulation was run, and its designs placed neighboring holes so
close that they overlapped. Those claims are withdrawn.

---

## Method

The design is an SSH ring: holes along a ring waveguide with alternating center-to-center spacings
`a` and `b`, hole radius `r`, ring radius `R` and waveguide width `w`.

- **Fabrication rule.** The gaps `a − 2r`, `b − 2r` and the edge clearance `(w − 2r)/2` must exceed the
  0.05 µm minimum feature size. The spacings stay in their original ranges, which keep the period
  `a + b` near what a band gap at 1.55 µm needs, so the holes are limited to radii of about
  0.03–0.07 µm.
- **Robustness.** Each design is scored over several realizations with random hole-radius disorder
  (4–8% standard deviation, depending on the config).
- **Single-objective search.** Gaussian-process Bayesian optimization (`gp_minimize`, `gp_hedge`
  acquisition) maximizing mean Q − 2 × Q standard deviation. Infeasible designs are scored 0 without
  simulating.
- **Multi-objective search.** NSGA-III over seven variables with four objectives (mean Q, Q standard
  deviation, bandgap, mode volume) and the fabrication rule as explicit constraints.
- **Reproducibility.** Every config sets `seed: 123`. Single-objective run directories store the full
  config, the config file name and the git commit; multi-objective runs store their config.

---

## Results

### Single-objective explorations

| Exploration | Evaluations | Best score | a (µm) | b (µm) | r (µm) | R (µm) | a/b |
|---|---|---|---|---|---|---|---|
| Extreme dimerization | 23 | 23,344 | 0.600 | 0.130 | 0.039 | 18.00 | 4.62 |
| Large rings | 23 | 14,987 | 0.450 | 0.175 | 0.048 | 15.00 | 2.57 |
| Strong dimerization (baseline) | 120 | 12,890 | 0.400 | 0.188 | 0.041 | 10.00 | 2.13 |
| Compact designs | 23 | 11,335 | 0.346 | 0.133 | 0.030 | 6.00 | 2.61 |
| Short test run | 15 | 11,175 | 0.400 | 0.135 | 0.030 | 12.00 | 2.96 |
| Fabrication limits | 30 | 11,094 | 0.400 | 0.200 | 0.040 | 8.00 | 2.00 |

Scores carry a constant −10,000 from the stand-in model's `r < 0.08 µm` penalty, which every
fabricable design triggers, so differences between them are meaningful but ratios are not.

![Parameter Space Exploration Results](images/parameter_exploration_comparison.png)

### Multi-objective Pareto front

2,000 designs evaluated, 1,867 feasible, 19 Pareto-optimal. Q-factors range from 15,297 to 37,361.
The front shows a direct Q-versus-bandgap trade-off: the five designs with Q above 34,000 have
bandgaps of 6.6–8.7, and the five with bandgaps above 20 have Q of 15,300–17,200. Recommended designs
for each objective are tabulated in the optimization report.

---

## Findings about the stand-in models

1. **Dimerization dominates the single-objective score.** The model adds 15,000 × (a − b)/(a + b);
   the extreme-dimerization run leads every other run by more than 8,000.
2. **Ring radius goes to a bound set by the period.** The net effect of R changes sign near
   a + b ≈ 0.63 µm, so the optimizers push R to whichever bound the period favors.
3. **Hole radius and waveguide width are nearly inert** within the fabricable range (Q changes by
   about 400 across r = 0.03–0.07 µm, below the disorder noise).
4. **The two models disagree on ring size.** The multi-objective model's radiation term
   `1 − exp(−5/R)` falls as R grows, so it favors small rings, contrary to its own code comment. The
   single-objective model has no such term.

These are properties of hand-written formulas. Testing whether any of them hold for real resonators
requires the MEEP integration.

---

## Next steps

1. **MEEP integration**: replace the stand-in Q with FDTD + Harminv, starting with a low-resolution
   config and a benchmark against the thesis's strong-dimerization design.
2. **Reconcile the stand-in models** (or retire them): the r < 0.08 µm penalty contradicts the
   fabrication rule, and the multi-objective radiation term has the wrong sign relative to its comment.
3. **Disorder applied twice** in the multi-objective path (8 outer × 8 inner realizations) should be
   reduced to one disorder model.

See [HANDOFF.md](HANDOFF.md) for the full list of open issues.

---

## Reproduce

```bash
python run_optimization.py --config configs/strong_dimerization_v1.yaml          # and the five explore/test configs
python run_multi_objective_optimization.py --config configs/multi_objective_v1.yaml
python compare_explorations.py
```

Results: `results/run_20261016_*/` and `results/multi_obj_20261016_015430/`.
