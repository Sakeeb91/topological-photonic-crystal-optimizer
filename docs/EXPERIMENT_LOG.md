# Experiment Log

The record of every simulation run for the SSH photonic-crystal cavity program. For each run it gives the code commit, the manifest, the cost, the results and the verdict, so another researcher can reproduce or extend any of them.

Background:
- the physics case, the literature and the plan are in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md);
- how to launch runs is in `docs/MODAL_RUNBOOK.md`, on branch `claude/modal-experiments`.

---

## Start here (new researcher)

### 1. Get the code

| Branch | PR | Contains |
|---|---|---|
| `claude/trusting-cerf-3qix4d` | [#4](https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/pull/4) | Literature review, pilot MEEP/MPB scripts, this log, the result snapshots in `research/results/`, and the Modal recipe in `CLAUDE.md` |
| `claude/modal-experiments` | [#5](https://github.com/Sakeeb91/topological-photonic-crystal-optimizer/pull/5) (based on #4) | The Modal app (`research/modal/`), experiment manifests E0-E5, `docs/MODAL_RUNBOOK.md`, and tests |

Until both PRs are merged, work from `claude/modal-experiments`; it contains everything in #4.

```bash
git clone https://github.com/Sakeeb91/topological-photonic-crystal-optimizer.git
```

```bash
cd topological-photonic-crystal-optimizer && git checkout claude/modal-experiments
```

### 2. Set up Modal (your own account)

Results are stored on a Modal Volume, `tpc-results`, in the owner's workspace (`sakeeb91`). **Another researcher's account cannot read it.** Use the snapshot in `research/results/` instead, or re-run the experiments: re-runs are cached by a hash of the job parameters, so identical parameters give identical result keys.

On a laptop:

```bash
pip install "modal[api-proxy-support]" pyyaml numpy
```

```bash
modal setup
```

In a Claude Code cloud session, follow the "Running on Modal" section of `CLAUDE.md`. It covers:
- the environment allowlist: `api.modal.com`, `input-plane.us-east.modal.com`, `clobber.modal-storage.com`;
- `SSL_CERT_FILE=/root/.ccr/ca-bundle.crt`;
- the token flow.

### 3. Reproduce or extend

```bash
modal run research/modal/app.py::e0
```

Run order: E0, E1, E2, E3, then E4 and E5. Each entry point accepts `--dry-run` (prints the jobs and an estimated cost, runs nothing) and `--manifest PATH`. `modal run research/modal/app.py::report --exp E3` rebuilds a verdict from the Volume.

### 4. Use the existing results without re-running

`research/results/modal_volume_2026-11-05/` is a byte-for-byte copy of the Volume taken on 2026-11-05.
- Layout: `<EXP>/<kernel>/<sha256(params)[:20]>.json`.
- Every file carries its full input (`job.params`), the solver output, `meep_version`, and the Modal tier and wall time.

---

## Status at a glance (2026-11-05)

| Exp | Purpose | Code commit | Jobs | Est. cost | Verdict |
|---|---|---|---|---|---|
| E0 | Smoke test: image, MPB and MEEP on Modal | `0bb888f` | 2 | <$0.01 | **go** |
| E1 | Bend-loss Q map; picks the platform | `7a60192` (re-scored at `d3966b7`) | 44 | ~$0.11 | **go** (after the scoring-rule fix) |
| E2 | Fabrication-constrained SSH gap maps | `7a60192` | 160 | ~$0.25 | **go** for Si 220; **no** for SiN 400 |
| E3 | 3D domain-wall Q, flux split, convergence | `9f828c4` | 18 | ~$2.86 | **no-go**: unconverged, mirror-limited, control untuned |
| E3-probe-40 | Same as E3 at res 40, Nside 30, T 600 (A and tanh walls, best cell) | `d3966b7` | 3 | ~$3.6 | running |
| E3-probe-60 | Same at res 60 | `d3966b7` | 2 | ~$18 | running (detached) |

Costs are the app's pre-run estimates. E3's actual wall times came in 2-4x shorter than estimated. Actual spend is shown on the Modal dashboard (workspace `sakeeb91`, app `tpc-experiments`).

---

## E0: smoke test (go)

- Manifest: `research/modal/manifests/e0.yaml`. Log: `research/results/logs/e0_run2.log`.
- MPB cell (sin400, oxide-filled holes, w = 1.0 um): gap 1540.6-1557.0 nm (16.4 nm).
- Bend Q of a bare ring (sin400, oxide, R = 10 um, w = 1.0 um, res 20, T = 300): 2543 at 1571.8 nm. **Unconverged:** E1 gives 2022 at res 30 and 1743 at res 45.
- Showed that the MPICH MEEP image builds and runs on Modal.

## E1: bend-loss map (go)

- Manifest: `e1.yaml`. Logs: `e1_run.log` and `e1_report2.log`.
- Exact 3D cylindrical MEEP with a bare ring and fixed azimuthal order m.

| Stack | Smallest R with Q_bend >= 1e5 | Notes |
|---|---|---|
| Si 220, oxide-clad, w 0.45-0.60 | **R = 2 um** (w=0.45: 2.9e5) | Q_bend > 1e6 for R >= 3: bend loss is irrelevant on Si |
| Si 220, SiO2 below / air top | R = 2 um (w=0.45: 9.6e5) | |
| SiN 400, oxide-clad, w = 0.8 | R = 30 um (4.1e6) | |
| SiN 400, oxide-clad, w = 1.0 | R = 20 um (4.7e5-6.5e5) | Likely multimode; check TE1 |
| SiN 400, oxide-clad, w = 1.2 | R = 15 um (1.4e5) | |

Convergence pairs, where "cap" means Q above 1e6 cannot be resolved at these run lengths:

| Pair | Q values | Verdict |
|---|---|---|
| Si w0.5 R3 | 1.2e9 vs 8.0e8 | agree, >cap |
| Si w0.5 R5 | 4.4e10 vs 1.9e8 | agree, >cap |
| SiN w1.0 R10 | 2022 vs 1743 | agree (16%) |
| SiN w1.0 R20 | 6.5e5 vs 4.7e5 | 37% apart, but both are above the 1e5 decision threshold |

The first scoring pass returned no-go because it demanded agreement between unresolvable Q values. The rule was fixed in `9f828c4` and `d3966b7` with no re-simulation.

## E2: fabrication-constrained SSH gap maps (go for Si, no for SiN)

- Manifest: `e2.yaml`. Log: `e2_run.log`.
- Rules: ANT Si 220 needs feature and web >= 60 nm; ANT SiN 400 needs >= 120 nm. Web = gap minus hole diameter.
- **SiN 400:** 0 of 40 cells can be fabricated. An SSH hole lattice at 1547 nm is not realizable on the ANT SiN MPW.
- **Si 220:** 36 of 120 cells are feasible. Best per stack, ranked by gap / mid-gap:

| Stack | a / b / r (um) | Gap | Mid-gap | Min web |
|---|---|---|---|---|
| air top (`sub`), w 0.5 | 0.2225 / 0.1335 / 0.0356 | 59.2 nm (3.8%) | 1555 nm | 62.3 nm |
| `sub`, w 0.6 | 0.1961 / 0.1449 / 0.0409 | 47.7 nm (3.1%) | 1559 nm | 63.1 nm |
| oxide-filled, w 0.5 | 0.2000 / 0.1478 / 0.0417 | 42.0 nm (2.7%) | 1554 nm | 64.4 nm |
| oxide-filled, w 0.6 | 0.1870 / 0.1382 / 0.0390 | 34.0 nm (2.2%) | 1554 nm | 60.2 nm |

- **Fabrication margin:** every feasible Si cell has a minimum web of 60-64 nm, at the edge of the rule. A tapeout design needs an extended E2 grid that targets a margin of at least 10 nm.

## E3: 3D domain-wall cavities (no-go; diagnostic)

- Manifest: `e3.yaml`. Log: `e3_run.log`.
- Setup: the two best `sub` w0.5 cells from E2, at 20 periods per side and T_harm = 400.
- Designs: abrupt walls A and B, a tanh-tapered wall (M = 6), and a Quan-Loncar trivial control.
- The table below is built from the raw files. The app's own E3 report keys results by design name only and mixes the two cells, so do not use its convergence numbers.

| design | cell | res | Q_total | Q_rad | Q_axial | lambda (um) | wall (min) | result file |
|---|---|---|---|---|---|---|---|---|
| quan_loncar | QL | 20 | 280 | 1.05e+04 | 287 | 1.6116 | 1.7 | `27bfe0af9c` |
| quan_loncar | QL | 20 | 345 | 1.36e+04 | 354 | 1.6242 | 1.1 | `03e156f882` |
| quan_loncar | QL | 30 | 337 | 1.29e+04 | 347 | 1.6153 | 3.4 | `b861efb249` |
| quan_loncar | QL | 30 | 264 | 1.11e+04 | 270 | 1.6006 | 6.2 | `ede9a7c1d8` |
| ssh_A | a=0.2070 r=0.0432 | 20 | 1.35e+03 | 4.28e+04 | 1.39e+03 | 1.5440 | 1.5 | `2c35b99bb4` |
| ssh_A | a=0.2070 r=0.0432 | 30 | 929 | 1.39e+05 | 935 | 1.5469 | 10.1 | `2eec725c53` |
| ssh_A | a=0.2225 r=0.0356 | 20 | 2.62e+03 | 8.11e+04 | 2.7e+03 | 1.5641 | 1.6 | `24fb4611be` |
| ssh_A | a=0.2225 r=0.0356 | 30 | 1.09e+03 | 1.18e+05 | 1.1e+03 | 1.5737 | 9.2 | `63e8b8a5d6` |
| ssh_B | a=0.2070 r=0.0432 | 20 | 1.42e+03 | 1.43e+04 | 1.57e+03 | 1.5260 | 1.5 | `3f063a560c` |
| ssh_B | a=0.2070 r=0.0432 | 30 | 974 | 3.45e+04 | 1e+03 | 1.5311 | 5.2 | `1f0cb3b64e` |
| ssh_B | a=0.2225 r=0.0356 | 20 | 2.22e+03 | 9.74e+03 | 2.87e+03 | 1.5327 | 1.4 | `74b90f5343` |
| ssh_B | a=0.2225 r=0.0356 | 30 | 1.26e+03 | 2.83e+04 | 1.32e+03 | 1.5475 | 5.6 | `460e6225ed` |
| ssh_tanh | a=0.2070 r=0.0432 | 20 | 950 | 5.01e+04 | 968 | 1.5315 | 1.2 | `34bea5a07e` |
| ssh_tanh | a=0.2070 r=0.0432 | 30 | 671 | 2.49e+05 | 673 | 1.5363 | 5.2 | `8166f30cb0` |
| ssh_tanh | a=0.2225 r=0.0356 | 20 | 2.04e+03 | 2.95e+05 | 2.05e+03 | 1.5422 | 1.5 | `ccf70ef968` |
| ssh_tanh | a=0.2225 r=0.0356 | 30 | 952 | 2.05e+05 | 956 | 1.5556 | 5.8 | `1dabbf6546` |

What it shows:
1. **Mirror-limited.** Q_total is about Q_axial, roughly 1e3, for every design. Leakage through 20-period mirrors hides Q_rad, which is the quantity of interest. Mirrors need lengthening until Q_axial >= 10 x Q_rad, or Q_rad should be compared directly.
2. **Unconverged, as expected for the hole size.** Within the same cell, res 20 to res 30 moves lambda by:
   - 9.6-14.8 nm for the r = 36 nm cell, where a hole is about 1.1 px at res 30;
   - 2.9-5.1 nm for the r = 43 nm cell.

   Q_rad changes by up to 5x. Larger holes converge faster, and 3D cost grows as res^4, so the selection should favour larger r/L.
3. **Q_rad ordering, unconverged** (res 30, A / B / tanh):

   | Cell | A | B | tanh |
   |---|---|---|---|
   | r = 36 nm | 1.2e5 | 2.8e4 | 2.1e5 |
   | r = 43 nm | 1.4e5 | 3.5e4 | 2.5e5 |

   The tapered wall gives the highest Q_rad and the B wall the lowest, in both cells. A and B differ by about 4x, consistent with the pilot's A/B inequivalence.
4. **The trivial control is not tuned.** It resonates at 1600-1624 nm, outside the window, with Q_rad of about 1.1e4-1.4e4. It needs its own design pass before any topological-vs-trivial comparison is fair.
5. Multi-rank `mpirun` (8 and 16 ranks) works on Modal. The opposite-parity check is clear for both cells: no TE1-like band sits inside the gap.

### E3 convergence probes (running)

- Manifests: `research/results/manifests/e3_probe40.yaml` and `e3_probe60.yaml`.
- Probe settings: only the best cell (`top_k: 1`), only the A and tanh walls, Nside 30, T_harm 600. Everything else follows `e3.yaml` at `d3966b7`.
- Purpose: a three-point resolution trend (res 20/30, then 40, then 60) to decide whether res >= 60 is needed for the 36 nm holes, or whether selection should move to larger-hole cells.
- Res 60 was launched with `--detach`, so it finishes even if the launching client disconnects. Its results land on the Volume and appear with `modal run research/modal/app.py::report --exp E3`.
- **Update this section and the snapshot when the probes finish.**

---

## Open issues (owner: the next person to touch the code)

1. The E3 report keys by design only. Key by (cell, design).
2. Retune the Quan-Loncar control to 1547 nm, then scan taper length.
3. Choose mirror length per design so that Q_axial >= 10 x Q_rad.
4. Resolution policy: hole radius >= 2 px, with convergence checked over 3 resolutions. Or bias E2 selection toward larger r.
5. Extended E2 grid with a fabrication margin of at least 10 nm, for any tapeout candidate.
6. E4 (disorder ensembles) waits until E3 is converged and the control is fair. Running E4 on the current settings would mostly measure numerical noise.

## Changelog

- **2026-11-05:** E0-E3 run; probes launched; Volume snapshot and logs committed under `research/results/`.
