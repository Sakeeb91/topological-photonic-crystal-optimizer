"""
Modal App for the SSH photonic-crystal experiments (docs/LITERATURE_REVIEW.md section 7).

    modal run research/modal/app.py::e0                 # smoke test, minutes, cents
    modal run research/modal/app.py::e1 [--dry-run]     # bend-loss map
    modal run research/modal/app.py::e2                 # fab-constrained MPB gap maps
    modal run research/modal/app.py::e3                 # 3D wall Q + flux split + res 20/30
    modal run research/modal/app.py::e4 --n 48          # matched disorder ensembles
    modal run research/modal/app.py::e5                 # unrolled-ring supercell + full ring
    modal run research/modal/app.py::report --exp E3    # go/no-go summary from the Volume

Every entrypoint takes --dry-run (expand the manifest, print jobs and cost, run nothing) and
--manifest PATH (use another YAML). Results are JSON files on the Volume `tpc-results`,
at /<EXP>/<kernel>/<sha256(params)[:20]>.json; jobs whose file exists are skipped.
See docs/MODAL_RUNBOOK.md.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, "/root/tpc"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import tpc_plan as P  # noqa: E402

APP_NAME = "tpc-experiments"
VOL_NAME = "tpc-results"
VOL_MOUNT = "/results"

# Same stack as research/pilot_sims/setup_env.sh (env "pmpi"), pinned to the versions it
# resolved on 2026-11-05: pymeep 1.35.0 / mpb 1.12.0 (MPICH builds), python 3.11.
image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "pymeep=1.35.0=mpi_mpich*",
        "mpb=1.12.0=mpi_mpich*",
        "mpich=4.3.2",
        "numpy=2.4.6",
        "scipy=1.17.1",
        "h5py=3.16.0",
        "pyyaml",
        channels=["conda-forge"],
    )
    .env({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "HYDRA_LAUNCHER": "fork"})
    .add_local_dir(HERE, "/root/tpc", ignore=["**/__pycache__/**", "results_local/**", "*.pyc"])
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


# ----------------------------------------------------------------------------
# Container side
# ----------------------------------------------------------------------------
def _run(job: dict, nproc: int) -> dict:
    rel = P.job_path(job)
    out_path = os.path.join(VOL_MOUNT, rel)
    vol.reload()
    if os.path.exists(out_path):
        with open(out_path) as f:
            r = json.load(f)
        return dict(key=job["key"], tag=job["tag"], cached=True, status=r.get("status", "ok"))
    with tempfile.TemporaryDirectory() as d:
        pin, pout = os.path.join(d, "in.json"), os.path.join(d, "out.json")
        with open(pin, "w") as f:
            json.dump(job["params"], f)
        cmd = [sys.executable, "/root/tpc/tpc_physics.py", job["kernel"], pin, pout]
        if nproc > 1:
            cmd = ["mpirun", "-np", str(nproc)] + cmd
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if proc.returncode != 0 or not os.path.exists(pout):
            fail = dict(job=job, returncode=proc.returncode, stderr_tail=proc.stderr[-4000:],
                        stdout_tail=proc.stdout[-2000:], wall_s=wall)
            fpath = os.path.join(VOL_MOUNT, job["exp"], "_failed", f"{job['key']}.json")
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "w") as f:
                json.dump(fail, f, indent=1)
            vol.commit()
            return dict(key=job["key"], tag=job["tag"], cached=False, status="failed", returncode=proc.returncode)
        with open(pout) as f:
            res = json.load(f)
    res["job"] = job
    res["modal"] = dict(tier=job["tier"], nproc=nproc, wall_s=wall, cpu=P.TIERS[job["tier"]]["cpu"],
                        memory_MiB=P.TIERS[job["tier"]]["memory"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, out_path)
    vol.commit()
    return dict(key=job["key"], tag=job["tag"], cached=False, status=res.get("status", "ok"), wall_s=wall)


def _kw(tier: str) -> dict:
    t = P.TIERS[tier]
    return dict(volumes={VOL_MOUNT: vol}, cpu=t["cpu"], memory=t["memory"], timeout=t["timeout"],
                retries=modal.Retries(max_retries=1, initial_delay=10.0))


@app.function(**_kw("light"))
def run_light(job: dict) -> dict:
    """MPB unit cells and cylindrical bend-loss Q: serial, 1 core, 2 GiB, 1 h."""
    return _run(job, P.TIERS["light"]["np"])


@app.function(**_kw("3d"))
def run_3d(job: dict) -> dict:
    """3D nanobeams up to ~4e6 reduced voxels: 8 cores, 8 GiB, 6 h, mpirun -np 8."""
    return _run(job, P.TIERS["3d"]["np"])


@app.function(**_kw("3d_big"))
def run_3d_big(job: dict) -> dict:
    """res-30 nanobeams and unrolled-ring supercells: 16 cores, 16 GiB, 12 h, mpirun -np 16."""
    return _run(job, P.TIERS["3d_big"]["np"])


@app.function(**_kw("3d_xl"))
def run_3d_xl(job: dict) -> dict:
    """Nanobeams above 1.5e7 reduced voxels (res >= 60 on small Si holes): 32 cores, 32 GiB, 24 h."""
    return _run(job, P.TIERS["3d_xl"]["np"])


@app.function(**_kw("ring"))
def run_ring(job: dict) -> dict:
    """Full 3D ring: 32 cores, 32 GiB, 24 h (Modal's maximum timeout), mpirun -np 32."""
    return _run(job, P.TIERS["ring"]["np"])


FN = {"light": run_light, "3d": run_3d, "3d_big": run_3d_big, "3d_xl": run_3d_xl, "ring": run_ring}


# ----------------------------------------------------------------------------
# Local side
# ----------------------------------------------------------------------------
def _vol_json(path: str):
    return json.loads(b"".join(vol.read_file(path)))


def _existing(exp: str) -> set[str]:
    try:
        ents = vol.listdir(f"/{exp}", recursive=True)
    except Exception:  # noqa: BLE001  (directory absent on first run)
        return set()
    return {os.path.basename(e.path)[:-5] for e in ents
            if e.path.endswith(".json") and "/_failed/" not in "/" + e.path and not e.path.endswith("summary.json")}


def load_results(exp: str) -> list[dict]:
    out = []
    try:
        ents = vol.listdir(f"/{exp}", recursive=True)
    except Exception:  # noqa: BLE001
        return out
    for e in ents:
        if e.path.endswith(".json") and "_failed" not in e.path and not e.path.endswith("summary.json"):
            out.append(_vol_json(e.path))
    return out


def _print_plan(jobs: list[dict]) -> None:
    s = P.summarize_cost(jobs)
    print(f"[plan] {s['n_jobs']} jobs by tier {s['by_tier']}; estimated {s['container_hours']:.2f} container-h, "
          f"~${s['usd']:.2f}, longest job {s['longest_job_hours']:.2f} h (estimates; see docs/MODAL_RUNBOOK.md)")
    for j in jobs[:12]:
        c = P.cost(j)
        print(f"   {j['key']} {j['kernel']:<11} {j['tier']:<6} {c['hours']*60:7.1f} min ${c['usd']:.3f}  {j['tag']}")
    if len(jobs) > 12:
        print(f"   ... {len(jobs) - 12} more")


def _spent_usd() -> float:
    """Program spend so far (apps in P.PROGRAM_APPS), hourly rows up to now + 1 h. Raises
    P.SpendUnknown when the total is zero or the report cannot be read: callers refuse (fail closed)."""
    return P.read_program_spend(modal.Workspace.from_context().billing)


def _dispatch(exp_name: str, manifest: str, dry_run: bool, n: int = 0, force: bool = False,
              budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0) -> list[dict]:
    man = P.load_manifest(manifest or exp_name)
    if n and man.get("ensemble"):
        man["ensemble"]["N"] = int(n)
    jobs = P.expand(man, load_results)
    _print_plan(jobs)
    done = _existing(man["experiment"])
    todo = [j for j in jobs if j["key"] not in done]
    too_long = [j for j in todo if P.cost(j)["hours"] > 0.8 * P.TIERS[j["tier"]]["timeout"] / 3600]
    if too_long and not force:
        print(f"[refuse] {len(too_long)} job(s) are estimated above 80% of their tier timeout (Modal max 24 h); "
              "they would likely be killed. Lower px_on_r / T_harm / Nside, select larger holes (select.min_r_nm), "
              "or pass --force. Running the rest.")
        for j in too_long[:6]:
            print(f"   {j['tag']}: est. {P.cost(j)['hours']:.1f} h on {j['tier']}")
        todo = [j for j in todo if j not in too_long]
    print(f"[skip] {len(jobs) - len(todo)} already on the Volume; {len(todo)} to run")
    if not todo:
        return []
    est = sum(P.cost(j)["usd"] for j in todo)
    if spent_usd < 0:
        try:
            spent_usd = _spent_usd()
            src = "workspace billing report"
        except Exception as e:  # noqa: BLE001  (SpendUnknown or an API error: fail closed)
            print(f"[budget] spend unknown ({type(e).__name__}: {e}). Refusing to launch; check "
                  "`modal billing report --for today --resolution h --json` and pass --spent-usd.")
            if not dry_run:
                return []
            spent_usd, src = float("nan"), "unknown"
    else:
        src = "--spent-usd"
    b = P.budget_check(spent_usd, est, budget_usd)
    print(f"[budget] spent ${b['spent_usd']:.2f} ({src}) + {P.BUDGET_SAFETY} x est ${est:.2f} = ${b['worst_case_usd']:.2f} "
          f"vs cap ${budget_usd:.2f}: {'OK' if b['ok'] else 'REFUSED'}")
    if dry_run:
        return []
    if not b["ok"]:
        print("[budget] refusing to launch. Reduce the manifest (fewer designs, ensemble.N: 0, lower px_on_r) or have "
              "the user raise the cap (--budget-usd).")
        return []
    out = []
    for tier in ("light", "3d", "3d_big", "3d_xl", "ring"):
        group = [j for j in todo if j["tier"] == tier]
        if not group:
            continue
        print(f"[run] {len(group)} x {tier} {P.TIERS[tier]}")
        for r in FN[tier].map(group, order_outputs=False, return_exceptions=True, wrap_returned_exceptions=False):
            if isinstance(r, BaseException):
                print(f"   ERROR {type(r).__name__}: {r}")
                out.append(dict(status="exception", error=str(r)))
            else:
                print(f"   {r['status']:<12} {r['key']} {r['tag']} {'(cached)' if r.get('cached') else ''}")
                out.append(r)
    return out


@app.local_entrypoint()
def e0(dry_run: bool = False, manifest: str = "", budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0):
    _dispatch("E0", manifest, dry_run, budget_usd=budget_usd, spent_usd=spent_usd)
    if not dry_run:
        print(json.dumps(P.gonogo("E0", load_results("E0")), indent=1))


@app.local_entrypoint()
def e1(dry_run: bool = False, manifest: str = "", budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0):
    _dispatch("E1", manifest, dry_run, budget_usd=budget_usd, spent_usd=spent_usd)
    if not dry_run:
        print(json.dumps(P.gonogo("E1", load_results("E1")), indent=1))


@app.local_entrypoint()
def e2(dry_run: bool = False, manifest: str = "", budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0):
    _dispatch("E2", manifest, dry_run, budget_usd=budget_usd, spent_usd=spent_usd)
    if not dry_run:
        print(json.dumps(P.gonogo("E2", load_results("E2")), indent=1))


@app.local_entrypoint()
def e3(dry_run: bool = False, manifest: str = "", force: bool = False, budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0):
    _dispatch("E3", manifest, dry_run, force=force, budget_usd=budget_usd, spent_usd=spent_usd)
    if not dry_run:
        print(json.dumps(P.gonogo("E3", load_results("E3")), indent=1))


@app.local_entrypoint()
def e4(dry_run: bool = False, manifest: str = "", n: int = 0, force: bool = False, budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0):
    """Disorder ensembles; --n overrides the manifest's realizations per cell (48 for the paper)."""
    _dispatch("E4", manifest, dry_run, n=n, force=force, budget_usd=budget_usd, spent_usd=spent_usd)
    if not dry_run:
        _summarize_e4(manifest, n)


@app.local_entrypoint()
def e5(dry_run: bool = False, manifest: str = "", budget_usd: float = P.BUDGET_USD, spent_usd: float = -1.0):
    _dispatch("E5", manifest, dry_run, budget_usd=budget_usd, spent_usd=spent_usd)
    if not dry_run:
        print(json.dumps(P.gonogo("E5", load_results("E5")), indent=1))


def _summarize_e4(manifest: str, n: int) -> None:
    man = P.load_manifest(manifest or "E4")
    if n:
        man["ensemble"]["N"] = int(n)
    jobs = {j["key"]: j for j in P.expand(man, load_results)}
    rows = [(r["job"]["key"], r) for r in load_results("E4") if r.get("job", {}).get("key") in jobs]
    summ = P.group_ensemble(rows, jobs, pt=dict(man.get("pt") or {}, classes=man["ensemble"]["classes"]) if man.get("pt") else None)
    with vol.batch_upload(force=True) as b:
        tmp = os.path.join(tempfile.mkdtemp(), "summary.json")
        with open(tmp, "w") as f:
            json.dump(summ, f, indent=1)
        b.put_file(tmp, "/E4/summary.json")
    print(json.dumps(summ, indent=1))


@app.local_entrypoint()
def spend():
    """Print program spend against the cap (apps in tpc_plan.PROGRAM_APPS since PROGRAM_START)."""
    try:
        s = _spent_usd()
    except Exception as e:  # noqa: BLE001
        print(f"program spend UNKNOWN ({type(e).__name__}: {e}); launches are refused until it can be read "
              "or --spent-usd is given")
        return
    print(f"program spend ${s:.2f} of ${P.BUDGET_USD:.2f} cap ({', '.join(P.PROGRAM_APPS)} since {P.PROGRAM_START})")


@app.local_entrypoint()
def report(exp: str = "E0"):
    """Print the go/no-go check for one experiment from results already on the Volume."""
    print(json.dumps(P.gonogo(exp.upper(), load_results(exp.upper())), indent=1))


if __name__ == "__main__" and "--local" in sys.argv:
    # Local backend (no Modal account, network or credits):  python research/modal/app.py --local E2 [...]
    import run_local
    run_local.main([a for a in sys.argv[1:] if a != "--local"])
