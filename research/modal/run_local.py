"""
Local backend: run experiment jobs in a local MEEP/MPB MPICH env instead of Modal, to save credits.

    S=$PWD/scratch bash research/pilot_sims/setup_env.sh
    python research/modal/app.py --local E2 [--manifest PATH] [--np 4] [--max-job-hours 0.5] [--dry-run]
    # or directly: python research/modal/run_local.py E2 ...

Same manifests, same job keys, same result layout as the Modal Volume:
    <out>/<EXP>/<kernel>/<sha256(params)[:20]>.json      (failures: <out>/<EXP>/_failed/<key>.json)
Default <out> is research/modal/results_local (git-ignored). Selections (E3-E5 pick cells from E2 results) read <out>
and the committed Volume snapshots research/results/modal_volume_*, so locally run stages can build
on results that were computed on Modal, and the other way round after uploading.

Runs: light-tier jobs (mpb_cell, bend_q, ql_period) always; 3D jobs only when the local cost model
(measured on the 4-core cloud container: 63 core-ns per reduced voxel-step at 2 ranks, 92 at 4)
estimates them below --max-job-hours. Larger jobs are listed and skipped: run them on Modal.
Unlike the Modal path this spends no credits, so there is no budget guard.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpc_plan as P  # noqa: E402

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEFAULT_OUT = os.path.join(HERE, "results_local")  # git-ignored; research/results/ is curated on the base branch
LOCAL_CORE_NS = {1: 63.0, 2: 63.0, 4: 92.0}  # measured on the 4-core cloud container (MODAL_RUNBOOK.md)


def local_hours(job: dict, nproc: int) -> float:
    e = P.estimate(job)
    if e["kind"] == "mpb":
        return 60 / 3600
    if e["kind"] == "cyl":
        return e["vox_reduced"] * e["steps"] * 6e-9 / 3600
    cns = LOCAL_CORE_NS.get(nproc, 100.0)
    return e["vox_reduced"] * e["steps"] * cns * 1e-9 / nproc / 3600


def find_env(explicit: str) -> str:
    if explicit:
        return explicit
    if os.environ.get("TPC_ENV"):
        return os.environ["TPC_ENV"]
    for cand in glob.glob(os.path.join(REPO, "**", "mamba", "envs", "pmpi"), recursive=True):
        return cand
    sys.exit("No local MEEP env: build one with research/pilot_sims/setup_env.sh and pass --env or set TPC_ENV "
             "to the 'pmpi' env prefix.")


def make_loader(dirs: list[str]):
    def load(exp: str) -> list[dict]:
        rows, seen = [], set()
        for d in dirs:
            for f in glob.glob(os.path.join(d, exp, "**", "*.json"), recursive=True):
                if "_failed" in f or f.endswith("summary.json"):
                    continue
                key = os.path.basename(f)[:-5]
                if key in seen:
                    continue
                seen.add(key)
                with open(f) as fh:
                    rows.append(json.load(fh))
        return rows
    return load


def existing_keys(dirs: list[str], exp: str) -> set[str]:
    return {os.path.basename(f)[:-5] for d in dirs for f in glob.glob(os.path.join(d, exp, "**", "*.json"), recursive=True)
            if "_failed" not in f and not f.endswith("summary.json")}


def run_job(job: dict, out: str, env: str, nproc: int) -> dict:
    path = os.path.join(out, P.job_path(job))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pin = path + ".in"
    with open(pin, "w") as f:
        json.dump(job["params"], f)
    n = 1 if job["tier"] == "light" else nproc
    cmd = [os.path.join(env, "bin", "python"), os.path.join(HERE, "tpc_physics.py"), job["kernel"], pin, path + ".tmp"]
    if n > 1:
        cmd = [os.path.join(env, "bin", "mpirun"), "-np", str(n)] + cmd
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    os.remove(pin)
    if pr.returncode != 0 or not os.path.exists(path + ".tmp"):
        fpath = os.path.join(out, job["exp"], "_failed", f"{job['key']}.json")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w") as f:
            json.dump(dict(job=job, returncode=pr.returncode, stderr_tail=pr.stderr[-4000:], wall_s=wall), f, indent=1)
        return dict(status="failed", key=job["key"], tag=job["tag"])
    with open(path + ".tmp") as f:
        res = json.load(f)
    res["job"] = job
    res["local"] = dict(backend="local", nproc=n, wall_s=wall, host_cpus=os.cpu_count())
    with open(path + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(path + ".tmp", path)
    return dict(status=res.get("status", "ok"), key=job["key"], tag=job["tag"], wall_s=wall)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("exp", help="E0..E5")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--env", default="")
    ap.add_argument("--np", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--max-job-hours", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=0, help="override ensemble.N (E4)")
    ap.add_argument("--limit", type=int, default=0, help="run at most this many jobs (0 = all eligible)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    exp = a.exp.upper()
    man = P.load_manifest(a.manifest or exp)
    if a.n and man.get("ensemble"):
        man["ensemble"]["N"] = a.n
    read_dirs = [a.out] + sorted(glob.glob(os.path.join(REPO, "research", "results", "modal_volume_*")), reverse=True)
    jobs = P.expand(man, make_loader(read_dirs))
    done = existing_keys(read_dirs, man["experiment"])
    todo = [j for j in jobs if j["key"] not in done]
    ok = [j for j in todo if j["tier"] == "light" or local_hours(j, a.np) <= a.max_job_hours]
    big = [j for j in todo if j not in ok]
    est = sum(local_hours(j, a.np) for j in ok)
    print(f"[local] {exp}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done (local or snapshot), "
          f"{len(ok)} to run here (~{est:.2f} h on {a.np} ranks), {len(big)} too large for local (> {a.max_job_hours} h each)")
    for j in big[:8]:
        print(f"   skip (run on Modal): {j['tag']}  est. {local_hours(j, a.np):.1f} h locally")
    if a.limit:
        ok = ok[: a.limit]
    if a.dry_run or not ok:
        return
    env = find_env(a.env)
    for j in ok:
        r = run_job(j, a.out, env, a.np)
        print(f"   {r['status']:<12} {j['kernel']:<11} {j['tag']}  {r.get('wall_s', 0):.0f}s")
    rows = make_loader(read_dirs)(man["experiment"])
    print(json.dumps(P.gonogo(man["experiment"], rows), indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
