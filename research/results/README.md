# Results snapshots

Raw outputs of the Modal experiments, committed so the results do not depend on access to the owner's Modal workspace.

- `modal_volume_2026-11-05/`: a copy of the Modal Volume `tpc-results`, taken with `modal volume get tpc-results / .` on 2026-11-05.
  - Layout: `<EXP>/<kernel>/<sha256(params)[:20]>.json`.
  - Each file holds the job parameters (`job.params`), the solver output, `meep_version`, and the Modal tier and wall time (`modal`).
  - Failed jobs, if any, are under `<EXP>/_failed/`.
- `manifests/`: manifests used for runs that are not in `research/modal/manifests/` (the E3 convergence probes).
- `logs/`: client-side run logs.
  - Box-drawing characters and local scratch paths have been stripped.
  - The `modal.com/apps/...` links point to the owner's dashboard.

Interpretation, verdicts and code commits for each run are in [`docs/EXPERIMENT_LOG.md`](../../docs/EXPERIMENT_LOG.md).

To refresh a snapshot after new runs:

```bash
modal volume get tpc-results / research/results/modal_volume_<DATE> --force
```

Then add an entry to the experiment log.
