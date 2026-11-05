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
- every pair in `resolution_convergence` has verdict `agree` or `same_side_of_threshold`, and `flagged_pairs` is empty.

How a pair is judged (`tpc_plan._gonogo_e1`). E1 reruns 4 rows at higher resolution (Si res 60, SiN res 45) with a 2x longer ringdown:
- **Q_cap = 1e6.** Harminv/FDTD cannot resolve bend Q above about 1e6-1e7 at these run lengths and PML settings. A pair where both members are above the cap agrees, and is reported as `>1e+06`.
- **Below the cap,** members agree if |dQ|/Q < 0.2.
- **`same_side_of_threshold`:** both members fall on the same side of 1e5, which is the only decision E1 makes.
- **Unresolved rows** (Q <= 0, or a change of more than 10% between Harminv at T/2 and at T) are handled as follows:
  - The pair agrees as "> cap" when neither member is a resolved below-cap value and at least one member is positive and above the cap. Example: Si R = 5 gave 4.4e10 (dQ_half 45) and 1.9e8 (dQ_half 7.3e3).
  - The pair is flagged when a resolved below-cap value meets an unresolved one, or when both members are unresolved and below the cap.

**First E1 run on Modal** (2026-11-05, 44 jobs, about 8 min, run by the parent session):

| Pair | Q (base res) | Q (high res) | Verdict |
|---|---|---|---|
| SiN 400, w = 1.0, R = 10 | 2022 (res 30) | 1743 (res 45) | agree, 16% |
| SiN 400, w = 1.0, R = 20 | 6.48e5 | 4.72e5 | 37%, same side of 1e5 |
| Si 220, w = 0.5, R = 3 | 1.22e9 | 7.99e8 | > cap |
| Si 220, w = 0.5, R = 5 | 4.4e10 (dQ_half 45) | 1.9e8 (dQ_half 7.3e3) | > cap |

What this means:
- E0's Q_bend = 2543 (res 20, T = 300) overestimated. The pilot's 1.6e3 was close to the converged value.
- **Si 220, oxide-clad:** Q_bend >= 2.9e5 already at R = 2 um (w = 0.45), and > 1e6 for R >= 3. **Bend loss is irrelevant on Si.**
- **SiN 400, oxide-clad:** Q_bend >= 1e5 needs w >= 1.2 at R = 15 (1.4e5), or w >= 1.0 at R = 20 (4.7e5 to 6.5e5). w = 0.8 needs R = 30.
- **Multimode warning:** SiN at w >= 1.0 um and 400 nm thickness is likely multimode.
  - Every E3/E4/E5 run now adds a `parity_check` MPB job per selected cell (`check_other_parity`). It reports opposite-parity (TE1/TM-like) zone-edge bands that fall inside the TE0 SSH gap.
  - Those bands do not couple in a y-symmetric beam, but transverse disorder (`offset_y`) mixes them.
  - Treat `other_parity_clear: false` as a no-go for that cell's offset_y ensemble.

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

**First E2 run on Modal** (2026-11-05, 160 jobs, run by the parent session; go = true):

SiN 400 (oxide-filled, w = 1.0 and 1.2): **0 of 40 cells pass the ANT 120 nm rules.** An SSH hole lattice at 1547 nm is not realizable on the ANT SiN MPW. This is a clean negative result in its own right.

Si 220: **36 feasible cells.** Best by gap-to-midgap ratio:

| Stack | Best cell (um) | Gap | Mid-gap | Min web |
|---|---|---|---|---|
| air top, w = 0.5 | a = 0.2225, b = 0.1335, r = 0.0356, L = 0.3561 | 59.2 nm (3.8%) | 1555 nm | 62.3 nm |
| air top, w = 0.6 | | 47.7 nm (3.1%) | | |
| oxide-filled, w = 0.5 | | 42.0 nm (2.7%) | | |
| oxide-filled, w = 0.6 | | 34.0 nm (2.2%) | | |

**Fabrication margin is the open risk.** Every feasible Si cell has a minimum web of 60-64 nm, right at the 60 nm rule. A few nm of global CD bias would break the rule.
- E2 now reports `fab_margin_nm` for each stack's best cell.
- Selections accept `select.min_margin_nm`, which requires that much margin above the rule on both web and feature.
- No current E2 cell has 10 nm of margin. A tapeout design (Paper B / C0) needs an extended E2 grid first: lower `delta` or smaller `r_over_L` near the gap/margin trade-off.
- E3 runs on the zero-margin pick; that is fine for physics, but not for a mask.

**E3: 3D wall Q.** E3 now runs in two stages. Both share the `/E3` folder, and `report --exp E3` judges them together.

```bash
modal run research/modal/app.py::e2 --manifest research/modal/manifests/e2b.yaml
```

```bash
modal run research/modal/app.py::e3 --dry-run
modal run research/modal/app.py::e3
```

```bash
modal run research/modal/app.py::e3 --manifest research/modal/manifests/e3c.yaml
```

The three runs are, in order:
1. **(Optional, recommended) E2b:** Si cells with larger holes and a fabrication margin. Cheap, and it cuts every 3D job below by about 4x (see "Cost problem").
2. **Stage 1, scan (`e3.yaml`):**
   - one resolution (1.5 px on the smallest hole radius);
   - Nside 20 / 30 / 40 for walls A, B and tanh;
   - a Quan-Loncar retune: `period_scale` 0.94 / 0.955 / 0.97, with `n_mirror` 15 / 25.
3. **Stage 2, convergence (`e3c.yaml`):**
   - first edit Nside / `period_scale` / `n_mirror` to the stage-1 picks;
   - three resolutions at 1.5 / 2 / 2.5 px on the smallest hole radius.

Go if every (cell, design) pair, including the Quan-Loncar control, has a variant that is:
- **converged** over at least 3 resolutions: at the top two, |dQ_rad|/Q_rad < 0.2 and |d lambda| < 2 nm, with at least 2 px on the smallest hole radius;
- **not mirror-limited:** Q_axial >= 10 x Q_rad;
- **resonant inside the source window.**

Compare designs on **Q_rad**, not Q_total. Results are keyed by (cell, design variant), so two selected cells no longer overwrite each other.

**First E3 run on Modal** (2026-11-05, 18 jobs from 9f828c4, run by the parent session; go = false):
- **Multi-rank `mpirun` works on Modal.**
- **Not converged.** Going from res 20 to res 30 moved lambda by 9.6 / 14.7 / 24.1 / 8.9 nm (A / B / tanh / Quan-Loncar), and Q by 140% (A) and 76% (B).
  - The cause: the selected Si holes (r = 36-43 nm) were only 0.7-1.1 px wide.
  - The old "px per wavelength" comment was the wrong criterion. **Resolution is now set from the smallest hole radius (`px_on_r`).**
- **Mirror-limited.** Every design had Q_total of about 1e3, with Q_axial of about 1e3, far below Q_rad (1e4 to 2e5).
  - At Nside = 20 the Q_total comparison measured mirror strength, not cavity quality.
  - Q_rad at res 30 (not converged): tanh 2.05e5, A 1.18e5, B 2.83e4, Quan-Loncar 1.29e4.
- **The Quan-Loncar control was not tuned.** It resonated at 1.615-1.624 um, outside the window, with Q_rad of 1.3e4.
  - Stage 1 retunes it with `period_scale`. The resonance needs to move about 4.5% to the blue, hence 0.955.
  - A topological-vs-trivial comparison is not fair until the control is tuned.

**Cost problem.** Resolving 36 nm holes at 2-2.5 px needs res 57-71, and cost scales as res^4.

Estimates (`tpc_plan.cost`, T_harm 1000 for E3 and 400 for E4):

| Run | E2 pick, r = 35.6 nm | Hypothetical cell, r = 50 nm |
|---|---|---|
| E3 stage 1 (16 jobs) | $42, 30 container-h | $9, 12 container-h |
| E3 stage 2 (13 jobs) | $169, 98 container-h, **3 jobs > 24 h** | $40, 31 container-h |
| E4, N = 4 | $421 | $94 |
| E4, N = 48 | **$4,940** | $1,100 |

The r = 50 nm column is a cost figure for that geometry only, not a claim that such a cell has a usable gap. E2b finds out.

What is now in place:
- Res >= ~60 routes to a new 32-core `3d_xl` tier, with a 24 h timeout.
- The dispatcher **refuses jobs estimated above 80% of their tier timeout** unless `--force` is passed. It prints what to change.

Cheaper routes, in order of value:
1. **Larger holes** (E2b, then `select.min_r_nm: 45` in e3/e4/e5.yaml). About 4x cheaper, and the same change gives the web margin a mask needs.
2. **E4 by perturbation theory, with FDTD only as validation.** Use first-order sensitivities d omega/d r_i and d omega/d x_i from one converged field (Johnson et al., PRE 65, 066611, 2002; review section 7). That gives frequency statistics for all four classes in seconds, with 5-10 FDTD runs per cell to check them. Not implemented yet; it is the next code step if E4 at N = 48 is wanted.
3. **GPU FDTD** (e.g. Tidy3D) for the ensemble, as the review suggested.

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
| 3d_big | 4e6-1.5e7 reduced voxels, supercells | 16 | 16 GiB | 12 h | 16 |
| 3d_xl | > 1.5e7 reduced voxels (res >= ~60 on Si holes) | 32 | 32 GiB | 24 h | 32 |
| ring | full 3D ring | 32 | 32 GiB | 24 h | 32 |

Memory is sized from the pilot full-ring run: 1.7 GB / 9.6e6 reduced voxels, about 200 B per voxel. That leaves about 5x headroom.

### Estimates

These are computed by `tpc_plan.cost()` from the manifests. For E3 to E5 they use a representative fabricable Si cell (a = 0.200, b = 0.160, r = 0.035 um, w = 0.5 um, air top). The real E2 pick is a = 0.2225, b = 0.1335, r = 0.0356 um; the E3/E4 rows use it, or a hypothetical r = 50 nm cell, and the stage-1/2 resolution rule.

| Exp | Jobs | Container-hours | Cost (USD) | Longest job | Wall clock |
|---|---|---|---|---|---|
| E0 | 2 | 0.03 | < 0.01 + image build | 2 min | ~10 min incl. first build |
| E1 | 44 | 1.7 | 0.11 | 10 min (res-60 convergence rows) | minutes |
| E2 | 160 | 4.0 | 0.25 | 2 min (flat MPB guess) | minutes |
| E3 (old plan, res 20/30) | 8 | 2.3 | 1.8 | 0.44 h | superseded: unconverged on Si holes |
| E3 stage 1 + 2, r = 35.6 nm | 29 | 128 | 211 | 20 h (3 jobs over the limit) | see "Cost problem" under E3 |
| E3 stage 1 + 2, r = 50 nm | 29 | 43 | 49 | 5 h | |
| E4, N = 48, r = 35.6 / 50 nm | 1156 | 2900 / 1035 | 4940 / 1100 | 5.7 / 1.4 h | use perturbation theory + FDTD spot checks |

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
