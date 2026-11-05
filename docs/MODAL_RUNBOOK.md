# Modal Runbook: running the SSH photonic-crystal experiments

**Date:** 2026-11-05
**Code:** [`research/modal/`](../research/modal/) (App `tpc-experiments`, Volume `tpc-results`)
**Science plan:** [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) sections 3.7, 5 and 7

## Status

Verified in this container:
- `app.py` parses and imports against `modal` 1.6.3.
  - It registers 4 functions and 7 local entrypoints.
  - `modal run research/modal/app.py::e0 --dry-run` gets as far as authentication and stops at `Token missing`.
- Every kernel (`mpb_cell`, `bend_q`, `nanobeam3d` with the 4 disorder classes, `supercell3d`, `ring3d`) ran here at validation scale and wrote valid JSON.
  - Validation scale means res 8-12 and short Harminv windows.
  - The stack was the same MEEP 1.35.0 / MPB 1.12.0 MPICH build the Modal image pins.
  - Fab refusal works: the pilot geometry (a/b/r = 0.35/0.15/0.10) is refused with `web -50.0 nm < min web 60 nm (8 of 16 webs)`.
- `pytest`: 108 passed, 13 of them new (`tests/test_modal_experiments.py`).

Verified on Modal (run by the parent session from the `topological-photonic-crystal-optimizer` environment, branch commit 0bb888f):
- E0 passed its go criterion with this app's MPICH image. That covers the image build, Volume writes, `.map()` and local Volume reads.
- E0 results: MPB cell (SiN 400, oxide-filled, w = 1.0) gap 1540.6-1557.0 nm (16.4 nm). Bend Q (R = 10, w = 1.0, oxide-clad) Q_bend = 2543 at 1571.8 nm, m = 67, 18.8 s. The bend value is not converged; see E1.

Not verified:
- **E1 to E5 have not run on Modal yet.** This session itself runs in an environment that blocks Modal. This session runs in an environment whose proxy refuses `api.modal.com:443` and `api.modal2.com:443` (policy 403). It also has no Modal credentials.
- `mpirun` with more than 1 rank inside a Modal container is untested: E0 jobs are single-rank. E3 is its first test.
- `CLAUDE.md` records a separate, verified Modal path from an earlier session: environment `topological-photonic-crystal-optimizer`, a `nompi` pymeep image, an E0 MPB cell that matched the local result. Section 3 follows that path. This app's image uses the **MPICH** build instead, which E0 will be the first to exercise on Modal.

No physics numbers are reported here. The validation runs used res 8-12, where Si holes are 1-2 pixels wide, and they say nothing about real Q.

---

## Your checklist, in order

### 1. Modal account: already set up

Modal is configured on your side (workspace `sakeeb91`; see `CLAUDE.md`). Nothing to do here.

Plan limits, for reference ([modal.com/pricing](https://modal.com/pricing)):

| Plan | Free credits per month | Seats | Containers |
|---|---|---|---|
| Starter | $30 | 3 | 100 |
| Team | $100 | unlimited | 5000 |

Academics can apply for credit grants of up to $10k. E0 to E3 plus E5 fit inside the Starter credit; a paper-scale E4 (N = 48) does not.

### 2. Optional: cap spend before E4 at N = 48

Do this on the Modal dashboard, under **Settings → Usage & Billing**. Only Workspace Owners or Managers can change it. [Source](https://modal.com/docs/guide/budgets).

There are two caps:
- **Workspace budget:** the hard cap on total usage for the billing cycle, counted before credits.
- **Spend limit:** the cap on net out-of-pocket charges after credits. When it is reached, Modal stops workloads that would cost you money.

Suggested settings:
- Start with a **$40 workspace budget**.
- Set a **$10 spend limit**.
- Raise both only when you get to E4 at N = 48.

### 3. Cloud-session setup (Claude does this; nothing for you to configure)

The recipe is from `CLAUDE.md`, commit 2a0edf4, verified on Modal on 2026-11-05.

**1. Environment.** The session must run in the cloud environment **`topological-photonic-crystal-optimizer`**:
- That environment allows `api.modal.com`, `input-plane.us-east.modal.com` (function calls) and `clobber.modal-storage.com` (Volume reads by the local entrypoint; needed by `_existing`, `load_results` and `report`).
- The Default environment blocks Modal.
- If the proxy refuses another Modal host, the refused host appears in `recentRelayFailures`; add that exact host to the environment. The likely candidate is `api.modal2.com`, the client's fail-over server.

```bash
curl -sS "$HTTPS_PROXY/__agentproxy/status"
```

**2. Client.** `ccr_client_setup.sh` installs `modal[api-proxy-support]`. Plain `modal` cannot use the HTTPS proxy and hangs.

```bash
bash research/modal/ccr_client_setup.sh
```

**3. TLS.** Every Modal command runs with `SSL_CERT_FILE` pointing at the agent proxy CA:

```bash
export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt
```

**4. Token.** `~/.modal.toml` does not persist across containers. When it is missing:
- Claude runs the command below in the background.
- Claude gives you the `https://modal.com/token-flow/...` link it prints, and you click it.
- Claude never asks for the token itself.

```bash
~/.venv-modal/bin/modal token new --source claude-code
```

### 4. Alternative: run from your own laptop

This path needs no change to the Claude Code environment. Python 3.10+ is enough locally; MEEP runs only in the Modal container.

```bash
git clone https://github.com/Sakeeb91/topological-photonic-crystal-optimizer.git
cd topological-photonic-crystal-optimizer
git checkout claude/modal-experiments
python3 -m venv .venv && source .venv/bin/activate
pip install modal pyyaml numpy
modal setup
```

### 5. Run E0 (smoke test)

What E0 does:
- It builds the image. Expect the first build to take several minutes (conda-forge solve plus downloads; not measured).
- It runs one MPB unit cell and one cylindrical ring.
- It writes both results to the Volume and prints a go/no-go block.

```bash
modal run research/modal/app.py::e0
```

**Go if:**
- the block says `"go": true`, meaning both kernels ran, the MPB gap is greater than 0, and Harminv found a bend mode;
- the bend Q for R = 10, w = 1.0 um, t = 0.4 SiN, oxide-clad lands near the pilot value of about 1.6e3 ([review section 4](LITERATURE_REVIEW.md#4-pilot-simulations-run-on-this-machine)). E0 runs it at res 20 with T = 300, so expect agreement within tens of percent.

**If it fails:**
- Open the Modal dashboard log for `run_light`.
- Check for a `/E0/_failed/<key>.json` on the Volume. It holds the stderr tail.

### 6. Run E1 → E5 with a go/no-go after each

Each command can be previewed first with `--dry-run`. That expands the manifest, prints every job with its estimated time and cost, and shows how many are already on the Volume. It runs nothing. Re-running any command skips jobs whose result JSON already exists.

**E1: bend-loss map.** Si 220 (w, R, oxide or air top) and SiN 400 (w, R).

```bash
modal run research/modal/app.py::e1
```

Go if:
- some (platform, w, cladding) reaches **Q_bend >= 1e5** at a usable R;
- for that row, `dQ_half` (the relative change between Harminv at T/2 and at T) is below 0.1;
- every pair in `resolution_convergence` agrees within |dQ|/Q < 0.2. E1 reruns 4 rows at higher resolution (Si res 60, SiN res 45) with a 2x longer ringdown for this.

Why the convergence check: on Modal, E0's res-20, T = 300 bend run (SiN, R = 10, w = 1.0) gave Q_bend = 2543. The pilot got 1.6e3 for the same cross-section at res 30 with T = 600-800. Low-resolution bend Q is not converged.

What to record:
- The output names the smallest such R per platform. Use it as R in `manifests/e5.yaml`.
- If SiN needs R >= 20 um, the SSH wall localization length (about 30 periods) is small relative to the ring. Treat SiN as the stretch platform, as the review recommends.

Watch the extrapolation: Harminv Q far above about 10 pi f T is extrapolated. Rows with Q above 1e7 are lower-confidence bounds, not measurements.

**E2: fabrication-constrained MPB gap maps.** The in-plane lattice is rescaled at fixed w and t until the gap is centred on 1547 nm.

```bash
modal run research/modal/app.py::e2
```

Go if: at least one cell on the chosen platform meets all three conditions:
- it passes that platform's fab rule (Si 60 nm, SiN 120 nm, on feature and web);
- its gap contains 1547 nm;
- it lies below the light line.

What the output shows:
- the feasible count per stack;
- the best cell;
- a count of which rule failed.

No-go:
- If nothing is feasible on SiN, that confirms the review's prediction that the 120 nm rule removes almost all dimerization. Proceed on Si only.
- If nothing is feasible on Si, stop. Widen the E2 grid (lower `delta`, larger `r_over_L`, other `w`) before spending on 3D.

**E3: 3D wall Q.** Runs walls A, B and tanh (M = 6), plus the Quan-Loncar trivial control. Each runs at res 20 and res 30, with a flux-box split into Q_rad and Q_axial (mirror and guided leakage).

```bash
modal run research/modal/app.py::e3 --dry-run
modal run research/modal/app.py::e3
```

Geometry is taken automatically from the best fabricable E2 cells: Si, `sub` cladding, top 2. Edit `select:` in `manifests/e3.yaml` to change it.

Go if:
- every design has a mode in the Harminv window;
- res 20 and res 30 agree: |dQ|/Q < 20% and |d lambda| < 2 nm;
- the Quan-Loncar control has a mode in the window.

If res 20 and 30 disagree, add `40` to `res:` and re-run; finished jobs are skipped. Si at res 20 is about 9 pixels per wavelength in Si, so this is likely.

If the control has no mode, tune `r_center_ratio` / `r_end_over_L` for `quan_loncar` until it does. E4 needs a working control.

Report for each design:
- `Q_total`, `Q_rad`, `Q_axial`, `frac_rad`;
- `T_harm`;
- `convergence.Q_over_pi_f_T`;
- `convergence.dQ_rel_half_vs_full`.

**E4: matched disorder ensembles.** Designs: abrupt SSH wall, tanh SSH wall and Quan-Loncar. Disorder classes, each in absolute nm with common random numbers: radius, longitudinal position, transverse offset, global CD bias. Sigma = 5 and 10 nm.

Start with the default N = 4, then go to N = 48:

```bash
modal run research/modal/app.py::e4
```

```bash
modal run research/modal/app.py::e4 --n 48
```

Go to N = 48 if:
- N = 4 finishes with fewer than 5% failed or empty-mode jobs;
- the per-cell spread looks sane.

The summary is written to `/E4/summary.json`. It holds, per (design, class, sigma) cell:
- harmonic-mean Q, Q10 and median Q;
- harmonic-mean Q_rad;
- sigma_lambda;
- 95% bootstrap intervals.

Caveat: 5 nm is about 0.1 pixel at res 20. MEEP's subpixel smoothing gives a smooth response, but its accuracy at that scale is a known limit (review section 4).

**E5 (optional): ring closure.** Runs unrolled-ring supercells (one A and one B wall, periodic, length = 2 pi R) and one full 3D ring at res 20.

```bash
modal run research/modal/app.py::e5
```

Go if:
- the supercell shows separate A and B wall modes (`primary_A`, `primary_B`);
- the full-ring Q is within about 3x of the Q budget 1/Q = 1/Q_bend (E1) + 1/Q_rad,wall (E3).

**Re-print any go/no-go block later:**

```bash
modal run research/modal/app.py::report --exp E3
```

### 7. Fetch results from the Volume

Layout: `/<EXP>/<kernel>/<sha256(params)[:20]>.json`. Each file holds the result, the full job spec, and the Modal tier and wall time. Failures go to `/<EXP>/_failed/`.

```bash
modal volume ls tpc-results /E3
```

```bash
modal volume get tpc-results /E3 ./results_modal/
```

```bash
modal volume get tpc-results /E4/summary.json ./results_modal/
```

---

## Expected cost and time

### Cost model

**Measured cost.** MEEP 1.35 MPICH on this container (Xeon 2.8 GHz, 4 cores), running `nanobeam3d` with flux monitors and two Harminv points at 6.84e5 reduced voxels, res 20:

| Ranks | ns per symmetry-reduced voxel-step | core-ns per voxel-step |
|---|---|---|
| 2 | 31.3 | 63 |
| 4 | 23.1 | 92 |

The parent review quotes about 10 ns at 4 ranks. That figure was not reproduced:
- The pilot logs show 27-35 ns at **2** ranks without flux monitors.
- The model therefore uses the measured 92 core-ns at 4 ranks.

**Formula.** hours = V_reduced × steps × 92 ns / (np × eff(np)), where:
- steps = (10/df + T_harm) × 2 × res. The first term is the Gaussian source duration; Courant number 0.5.
- eff(4) = 1.

**Assumptions, unmeasured:**
- Parallel efficiency beyond 4 ranks: eff(8) = 0.8, eff(16) = 0.65, eff(32) = 0.5. Efficiency from 2 to 4 ranks on this small cell was already 68%.
- One Modal "physical core" (2 vCPU, [docs](https://modal.com/docs/guide/resources)) runs one MPI rank at the speed of one core here.
- **Treat every 3D figure below as uncertain by a factor of 2 either way.**

**Pricing** ([modal.com/pricing](https://modal.com/pricing), 2026-11-05):
- CPU: $0.0000131 per physical core per second.
- Memory: $0.00000222 per GiB per second.
- Billing uses the larger of the request and actual use ([docs](https://modal.com/docs/guide/resources)).
- Volumes: $0.09 per GiB per month, with 1 TiB per month included.

**Limits:**
- Function timeout: at most 24 h ([docs](https://modal.com/docs/guide/timeouts)).
- CPU request: at most 64 cores. That number comes from a Modal error message quoted in [this project's notes](https://github.com/Open-Athena/MarinMat/blob/main/specs/31-cache-build-modal-cpu.md). Modal's own [resources page](https://modal.com/docs/guide/resources) says only that a maximum exists.
- Memory: the per-container maximum is not published.

### Tiers (`research/modal/tpc_plan.py: TIERS`)

| Tier | Used for | cpu (cores) | memory | timeout | MPI ranks |
|---|---|---|---|---|---|
| light | `mpb_cell`, `bend_q` | 1 | 2 GiB | 1 h | 1 |
| 3d | nanobeams <= 4e6 reduced voxels | 8 | 8 GiB | 6 h | 8 |
| 3d_big | res-30 nanobeams, supercells | 16 | 16 GiB | 12 h | 16 |
| ring | full 3D ring | 32 | 32 GiB | 24 h | 32 |

Memory is sized from the pilot full-ring run: 1.7 GB / 9.6e6 reduced voxels, about 200 B per voxel. That leaves about 5x headroom.

### Estimates

These are computed by `tpc_plan.cost()` from the manifests. For E3 to E5 they use a representative fabricable Si cell (a = 0.200, b = 0.160, r = 0.035 um, w = 0.5 um, air top). The real E2 pick will differ.

| Exp | Jobs | Container-hours | Cost (USD) | Longest job | Wall clock |
|---|---|---|---|---|---|
| E0 | 2 | 0.03 | < 0.01 + image build | 2 min | ~10 min incl. first build |
| E1 | 44 | 1.7 | 0.11 | 10 min (res-60 convergence rows) | minutes |
| E2 | 160 | 4.0 | 0.25 | 2 min (flat MPB guess) | minutes |
| E3 | 8 | 2.3 | 1.8 | 0.44 h (res 30) | ~0.5 h |
| E4, N = 4 | 99 | 18 | 7.8 | 0.28 h | ~0.5 h |
| E4, N = 48 | 1155 | 206 | 91 | 0.28 h | ~3-4 h at 100 containers |
| E5 | 3 | 2.4 | 3.0 | 1.0 h (full ring) | ~1 h |
| **Total (E4 at N = 48)** | | **~216** | **~$100** | | |

**What the cost scales with.**

Harminv window:
- T_harm = 400 resolves Q up to roughly 10 pi f T ≈ 8e3 without heavy extrapolation.
- Designs with Q of 1e5 need T of about 5000, which costs about 10x more per job. Check `Q_over_pi_f_T` in E3 before scaling E4.

Resolution:
- Cost goes as res^4.
- Moving E4 to res 30 multiplies it by about 5.

Free credit:
- Starter's $30 covers everything except paper-scale E4.
- E4 at N = 48 needs about $90 of credit or an academic grant.

---

## Code map

| File | Role |
|---|---|
| `research/modal/app.py` | Modal App: micromamba image (conda-forge `pymeep=1.35.0=mpi_mpich*`, `mpb=1.12.0=mpi_mpich*`, `mpich=4.3.2`, numpy/scipy/h5py, python 3.11), Volume `tpc-results`, functions `run_light` / `run_3d` / `run_3d_big` / `run_ring` (each runs `mpirun -np <cores>`), entrypoints `e0`-`e5`, `report` |
| `research/modal/tpc_physics.py` | Kernels: `mpb_cell`, `bend_q`, `nanobeam3d`, `supercell3d`, `ring3d`. MEEP/MPB only, no `modal` import |
| `research/modal/tpc_geometry.py` | SSH wall none/A/B/tanh, Quan-Loncar Gaussian mirror, unrolled-ring supercell, fab rules, CRN disorder |
| `research/modal/tpc_plan.py` | Manifest expansion, E2 → E3/E4/E5 selection, hash keys, tier routing, cost model, ensemble statistics, go/no-go checks |
| `research/modal/manifests/e0.yaml` … `e5.yaml` | Experiment definitions |
| `research/modal/validate_local.py` | Runs every experiment at validation scale with a local MEEP env, with no Modal |
| `research/modal/ccr_client_setup.sh` | Modal client setup inside a Claude Code cloud session (proxy extra, SSL_CERT_FILE, token flow) |
| `tests/test_modal_experiments.py` | Geometry, fab rules, CRN, keys, manifest expansion, statistics |

### Physics rules the code enforces

**Not reused from the repo or the pilot:**
- The repo's `src/simulation_wrapper.py` is not used. Its geometry has no domain wall, and its Q is a mock.
- The pilot 3D geometries are refused by the fab check. Their webs run from +10 nm down to -50 nm, and a test asserts the refusal.

**Fab rules** (`tpc_geometry.PLATFORMS`):

| Platform | Min feature | Min web |
|---|---|---|
| ANT Si 220 nm | 60 nm | 60 nm |
| ANT SiN 400 nm | 120 nm | 120 nm |

- Web = centre-to-centre gap - r_i - r_j. That is the gap minus the hole diameter when radii are equal.
- The side rail must be at least the larger of the two rules.
- 3D kernels refuse failing geometry: status `refused_fab`, with the reason logged and no simulation run.

**Disorder** (`tpc_geometry.apply_disorder`):
- Four classes, in absolute nm: radius, longitudinal position, transverse offset, global CD bias.
- Each class has its own random stream, keyed by (seed, realization, class) and indexed by a signed hole ordinal. A given hole gets the same draw in every design (common random numbers), and switching one class on leaves the others unchanged.
- Transverse disorder turns off the y-mirror symmetry, which costs 2x.

**Q reporting:**
- Q_total comes from Harminv.
- The split into Q_rad and Q_axial comes from DFT flux monitors added after the source ends: a closed box, plus windows of (w + 1) × (t + 1) um on the two end faces.
- Q_i = Q_total × P_total / P_i.
- Harminv runs to T/2 and to T; both results and the run length are stored.
- E3 runs res 20 and res 30.

---

## Sources

- Modal pricing, plans, credits, Sandbox multiplier: https://modal.com/pricing
- Modal CPU/memory requests, physical cores, soft CPU limit, billing on request: https://modal.com/docs/guide/resources
- Modal budgets and spend limits: https://modal.com/docs/guide/budgets
- Modal timeouts (24 h maximum): https://modal.com/docs/guide/timeouts
- Modal preemption, retries and `nonpreemptible` multiplier: https://modal.com/docs/guide/functions.md
- Modal images and `micromamba_install`: https://modal.com/docs/guide/images and https://modal.com/docs/reference/modal.Image
- 64-core per-function CPU ceiling (third-party report of Modal's error text): https://github.com/Open-Athena/MarinMat/blob/main/specs/31-cache-build-modal-cpu.md
- Modal client 1.6.3 source (`modal/config.py`, `modal/_utils/proxy_support.py`, `modal/_utils/blob_utils.py`, `grpclib/client.py`), read from the installed package
- Claude Code cloud environments, network access and environment variables: https://code.claude.com/docs/en/cloud-environments#network-access
- Verified cloud-session Modal recipe (environment, proxy extra, SSL_CERT_FILE, token flow): `CLAUDE.md`, commit 2a0edf4
- Quan and Loncar, "Deterministic design of wavelength scale, ultra-high Q photonic crystal nanobeam cavities", *Opt. Express* 19, 18529 (2011)
