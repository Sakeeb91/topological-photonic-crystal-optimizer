# Modal experiments (E0-E5)

Serverless runs of the experiment program in [`docs/LITERATURE_REVIEW.md`](../../docs/LITERATURE_REVIEW.md) section 7.
The step-by-step checklist, cost table and go/no-go criteria are in [`docs/MODAL_RUNBOOK.md`](../../docs/MODAL_RUNBOOK.md).

```bash
pip install modal pyyaml numpy && modal setup
modal run research/modal/app.py::e0
```

| Entrypoint | Manifest | What |
|---|---|---|
| `e0` | `manifests/e0.yaml` | smoke test: one MPB cell, one cylindrical ring |
| `e1` | `manifests/e1.yaml` | bend-loss Q map, Si 220 and SiN 400 |
| `e2` | `manifests/e2.yaml` | fab-constrained MPB SSH gap maps, gap rescaled onto 1547 nm |
| `e3` | `manifests/e3.yaml` | 3D wall Q (A, B, tanh, Quan-Loncar), flux split, res 20 vs 30 |
| `e4` | `manifests/e4.yaml` | matched CRN disorder ensembles (`--n 48` for paper scale) |
| `e5` | `manifests/e5.yaml` | unrolled-ring supercell + one full 3D ring |
| `report --exp EX` | | go/no-go block from results on the Volume |

All entrypoints take `--dry-run` and `--manifest PATH`.

Results are written to the Volume `tpc-results` at `/<EXP>/<kernel>/<hash>.json`; completed jobs are skipped.

## Without Modal

```bash
S=$PWD/scratch bash research/pilot_sims/setup_env.sh
TPC_ENV=$PWD/scratch/mamba/envs/pmpi python research/modal/validate_local.py --out /tmp/tpc_val
python research/modal/tpc_plan.py E0 E1 E2          # offline job count and cost estimate
```

`validate_local.py` runs every experiment at validation scale (res 8-12, short Harminv windows). Its numbers only show that the code runs; they are not physics.
