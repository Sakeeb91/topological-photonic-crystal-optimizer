"""
Local, Modal-free validation: expand every manifest, shrink each job to validation scale
(tiny resolution, short Harminv window, short chains, N=1), run the same tpc_physics.py
kernels with the local MPICH env, and write JSON with the same keys/layout as the Volume.

    S=<scratch> bash research/pilot_sims/setup_env.sh          # builds $S/mamba/envs/pmpi
    TPC_ENV=$S/mamba/envs/pmpi python research/modal/validate_local.py [--out DIR] [--np 4] [E0 E1 ...]

Numbers produced here are VALIDATION-SCALE ONLY (res 6-16): they prove the code runs and
writes valid JSON. They are not physics results.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tpc_plan as P  # noqa: E402

SMALL = {
    "mpb_cell": dict(res=10, Ly=2.0, Lz=2.0, rescale_iters=2),
    "bend_q": dict(res=12, T_after=60, neff_res=16),
    "nanobeam3d": dict(T_harm=60, nfreq=41),   # res comes from shrink_manifest
    "sens3d": dict(T_harm=60, n_theta=16, n_z=4),
    "supercell3d": dict(T_harm=40),
    "ring3d": dict(T_harm=20),
}
LIMIT = {"E0": 99, "E1": 2, "E2": 6, "E3": 8, "E4": 9, "E5": 2}


def shrink_manifest(man: dict) -> dict:
    m = copy.deepcopy(man)
    if "res" in m or "px_on_r" in m:
        m.pop("px_on_r", None)
        m["res"] = [10, 12]
    for d in m.get("designs", []):
        if "Nside" in d:
            d["Nside"] = 4
        if d["kind"] == "quan_loncar":
            d["n_taper"], d["n_mirror"] = 3, 3
            d.pop("period_scale", None)
    if "ensemble" in m:
        m["ensemble"]["N"] = 1
        m["ensemble"]["sigmas_nm"] = m["ensemble"]["sigmas_nm"][:1]
        m["res"] = [10]
    for sc in m.get("supercells", []):
        sc["R_um"], sc["res"] = 1.0, 8
    m["supercells"] = m.get("supercells", [])[:1]
    for rg in m.get("rings", []):
        rg["R_um"], rg["res"] = 1.2, 8
    if m.get("select"):
        m["select"]["top_k"] = 1
        m["select"].pop("min_r_nm", None); m["select"].pop("min_margin_nm", None)  # local E2 is tiny
    return m


def shrink_job(j: dict) -> dict:
    p = copy.deepcopy(j["params"])
    p.update(SMALL[j["kernel"]])
    if j["kernel"] == "bend_q":
        p["R"] = min(p["R"], 10.0)
    return P.make_job(j["exp"], j["kernel"], p, j["tag"] + "|validation")


def load_local(out: str):
    def f(exp: str):
        rows = []
        d = os.path.join(out, exp)
        for root, _, files in os.walk(d):
            if "_failed" in root:
                continue
            for fn in files:
                if fn.endswith(".json") and fn != "summary.json":
                    rows.append(json.load(open(os.path.join(root, fn))))
        return rows
    return f


def run(j: dict, out: str, env: str, nproc: int) -> dict:
    path = os.path.join(out, P.job_path(j))
    if os.path.exists(path):
        return dict(cached=True, **json.load(open(path)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pin = path + ".in"
    json.dump(j["params"], open(pin, "w"))
    py = os.path.join(env, "bin", "python")
    cmd = [py, os.path.join(HERE, "tpc_physics.py"), j["kernel"], pin, path]
    n = 1 if j["tier"] == "light" else nproc
    if n > 1:
        cmd = [os.path.join(env, "bin", "mpirun"), "-np", str(n)] + cmd
    t0 = time.time()
    pr = subprocess.run(cmd, capture_output=True, text=True)
    os.remove(pin)
    if pr.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f"{j['kernel']} {j['tag']} failed:\n{pr.stderr[-3000:]}")
    r = json.load(open(path))
    r["job"] = j; r["validation_scale"] = True; r["local"] = dict(nproc=n, wall_s=time.time() - t0)
    json.dump(r, open(path, "w"), indent=1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="*", default=["E0", "E1", "E2", "E3", "E4", "E5"])
    ap.add_argument("--out", default=os.path.join(HERE, "results_local"))
    ap.add_argument("--np", type=int, default=4)
    ap.add_argument("--env", default=os.environ.get("TPC_ENV", ""))
    a = ap.parse_args()
    if not a.env:
        sys.exit("set TPC_ENV to the pmpi env prefix built by research/pilot_sims/setup_env.sh")
    loader = load_local(a.out)
    report = {}
    for exp in a.exps:
        man = shrink_manifest(P.load_manifest(exp))
        full = P.expand(P.load_manifest(exp), loader)
        jobs = [shrink_job(j) for j in P.expand(man, loader)][: LIMIT[exp]]
        print(f"== {exp}: manifest expands to {len(full)} production jobs; running {len(jobs)} at validation scale")
        rows = []
        for j in jobs:
            r = run(j, a.out, a.env, a.np)
            rows.append(r)
            pm = (r.get("convergence") or {}).get("primary") or r.get("primary_A") or {}
            print(f"   {r.get('status', 'ok'):<12} {j['kernel']:<11} {j['tag']:<45} "
                  f"Q={pm.get('Q', r.get('Q_total'))} lam={pm.get('lam_um')} fab_ok={(r.get('fab') or r.get('fab_nominal') or {}).get('ok')} "
                  f"wall={r.get('kernel_wall_s', 0):.1f}s {'(cached)' if r.get('cached') else ''}")
        g = P.gonogo(exp, rows)
        report[exp] = dict(n_production_jobs=len(full), n_validated=len(rows),
                           statuses=sorted({r.get("status", "ok") for r in rows}), gonogo_go=g.get("go"))
        if exp == "E4":
            jobs_by_key = {j["key"]: j for j in jobs}
            summ = P.group_ensemble([(r["job"]["key"], r) for r in rows], jobs_by_key,
                                    pt=dict(man.get("pt") or {}, classes=man["ensemble"]["classes"], n_draws=200) if man.get("pt") else None)
            json.dump(summ, open(os.path.join(a.out, "E4", "summary.json"), "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
