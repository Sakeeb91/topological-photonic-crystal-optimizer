"""
Manifest expansion, result keys, resource tiers, cost estimates and ensemble statistics.

Pure Python + numpy + PyYAML; imported by app.py (locally and in the container) and by
validate_local.py. No MEEP, no modal.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
from typing import Any, Callable

import numpy as np

import tpc_geometry as G

CODE_VERSION = "1"  # bump to invalidate every cached result
HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Resource tiers. cpu = Modal physical cores (= 2 vCPU); np = MPI ranks.
# --------------------------------------------------------------------------
TIERS: dict[str, dict[str, Any]] = {
# Memory is billed on the request, so it is sized from BYTES_PER_REDUCED_VOXEL with ~5x headroom
# for the largest job each tier receives (route() caps 3d at 4e6 reduced voxels = 0.8 GB).
    "light":  dict(cpu=1.0,  memory=2048,  timeout=3600,      np=1),
    "3d":     dict(cpu=8.0,  memory=8192,  timeout=6 * 3600,  np=8),
    "3d_big": dict(cpu=16.0, memory=16384, timeout=12 * 3600, np=16),
    "ring":   dict(cpu=32.0, memory=32768, timeout=24 * 3600, np=32),  # 24 h is Modal's maximum
}

# MEEP 1.35 MPICH cost, measured 2026-11-05 on the 4-core cloud container (Xeon 2.8 GHz) with
# tpc_physics.nanobeam3d (flux monitors + 2 Harminv points on), 6.84e5 reduced voxels:
#   np=2: 31.3 ns per reduced voxel-step (63 core-ns);  np=4: 23.1 ns (92 core-ns).
# Cost per job = V_red * steps * CORE_NS / (np * PAR_EFF[np]), relative to np=4.
# PAR_EFF beyond 4 ranks is ASSUMED (unmeasured); see docs/MODAL_RUNBOOK.md.
CORE_NS_PER_VOXEL_STEP = 92.0
PAR_EFF = {1: 1.45, 2: 1.45, 4: 1.0, 8: 0.8, 16: 0.65, 32: 0.5}
BYTES_PER_REDUCED_VOXEL = 200.0  # from the pilot full-ring run: 1.7 GB / 9.6e6 voxels
CPU_USD_PER_CORE_S = 0.0000131   # modal.com/pricing, 2026-11-05
MEM_USD_PER_GIB_S = 0.00000222


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------
def job_key(kernel: str, params: dict) -> str:
    blob = json.dumps({"kernel": kernel, "params": params, "v": CODE_VERSION}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def job_path(job: dict) -> str:
    return f"{job['exp']}/{job['kernel']}/{job['key']}.json"


def make_job(exp: str, kernel: str, params: dict, tag: str = "") -> dict:
    params = copy.deepcopy(params)
    j = dict(exp=exp, kernel=kernel, params=params, tag=tag, key=job_key(kernel, params))
    j["tier"] = route(j)
    return j


# --------------------------------------------------------------------------
# Manifests
# --------------------------------------------------------------------------
def load_manifest(name_or_path: str) -> dict:
    import yaml
    p = name_or_path
    if not os.path.exists(p):
        p = os.path.join(HERE, "manifests", f"{name_or_path.lower()}.yaml")
    with open(p) as f:
        return yaml.safe_load(f)


def _set(d: dict, dotted: str, v):
    ks = dotted.split(".")
    for k in ks[:-1]:
        d = d.setdefault(k, {})
    d[ks[-1]] = v


def expand_grid(base: dict, sweep: dict | None) -> list[dict]:
    if not sweep:
        return [copy.deepcopy(base)]
    keys = list(sweep)
    out = []
    for combo in itertools.product(*[sweep[k] for k in keys]):
        p = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            if isinstance(v, dict) and "." not in k and isinstance(p.get(k), dict):
                p[k].update(v)
            else:
                _set(p, k, v)
        out.append(p)
    return out


def _grids(man: dict, exp: str) -> list[dict]:
    jobs = []
    for j in man.get("jobs", []) or []:
        jobs.append(make_job(exp, j["kernel"], j["params"], j.get("tag", "")))
    for g in man.get("grids", []) or []:
        for p in expand_grid(g["base"], g.get("sweep")):
            jobs.append(make_job(exp, g["kernel"], p, g.get("tag", "")))
    return jobs


def select_cells(sel: dict, results: list[dict]) -> list[dict]:
    """Pick fabricable MPB cells (E2 results) whose gap contains the target."""
    rows = [r for r in results if r.get("kernel") == "mpb_cell"]
    for k in ("platform", "clad", "w"):
        if k in sel:
            rows = [r for r in rows if r.get(k) == sel[k]]
    req = sel.get("require", ["fab_ok", "gap_contains_target", "below_lightline"])
    if "fab_ok" in req:
        rows = [r for r in rows if r.get("fab", {}).get("ok")]
    if "gap_contains_target" in req:
        rows = [r for r in rows if r.get("gap_contains_target")]
    if "below_lightline" in req:
        rows = [r for r in rows if r.get("below_lightline")]
    rows.sort(key=lambda r: -r.get(sel.get("rank_by", "gap_mid_ratio"), 0))
    return rows[: int(sel.get("top_k", 1))]


def _window(cell: dict, man: dict) -> tuple[float, float]:
    fmid = 0.5 * (cell["edge_f1"] + cell["edge_f2"])
    df = max(float(man.get("df_gap_multiple", 3.0)) * cell["gap_abs"], float(man.get("df_min_rel", 0.03)) * fmid)
    return fmid, df


def _cell_geometry(cell: dict) -> dict:
    return dict(platform=cell["platform"], w=cell["w"], clad=cell["clad"], a=cell["a"], b=cell["b"], r=cell["r"], L=cell["L"])


def _ql_design(cell: dict, ql: dict, plat: dict) -> dict:
    """Quan-Loncar trivial control at the same period and platform. r_end is the largest
    radius the web rule allows (minus a margin), r_center = ratio * r_end."""
    L = cell["L"]
    r_max = 0.5 * (L - plat["min_web"]) - 1e-3 * float(ql.get("margin_nm", 5))
    r_end = min(r_max, float(ql.get("r_end_over_L", 0.3)) * L)
    r_c = max(float(ql.get("r_center_ratio", 0.7)) * r_end, 0.5 * plat["min_feature"] + 1e-3 * float(ql.get("margin_nm", 5)))
    return dict(kind="quan_loncar", period=L, r_center=r_c, r_end=r_end,
                n_taper=int(ql.get("n_taper", 10)), n_mirror=int(ql.get("n_mirror", 10)))


def _designs_for_cell(cell: dict, man: dict) -> list[dict]:
    plat = G.platform(cell["platform"])
    out = []
    for d in man.get("designs", []):
        if d["kind"] == "ssh":
            dd = dict(kind="ssh", a=cell["a"], b=cell["b"], r=cell["r"], Nside=int(d.get("Nside", 20)), wall=d["wall"])
            if d["wall"] == "tanh":
                dd["M"] = float(d.get("M", 6))
            out.append((d.get("name", f"ssh_{d['wall']}"), dd))
        elif d["kind"] == "quan_loncar":
            out.append((d.get("name", "quan_loncar"), _ql_design(cell, d, plat)))
    return out


def expand(man: dict, load_results: Callable[[str], list[dict]] | None = None) -> list[dict]:
    exp = man["experiment"]
    jobs = _grids(man, exp)
    sel = man.get("select")
    if not sel:
        return jobs
    if load_results is None:
        raise RuntimeError(f"{exp} selects geometry from {sel['from']} results; none available")
    cells = select_cells(sel, load_results(sel["from"]))
    if not cells and man.get("fallback_cells"):
        cells = man["fallback_cells"]
    sim = man.get("sim", {})
    for cell in cells:
        geo = _cell_geometry(cell)
        fcen, df = _window(cell, man)
        base = dict(platform=geo["platform"], w=geo["w"], clad=geo["clad"], fcen=fcen, df=df, **sim)
        kind = man.get("kind", "nanobeam")
        if kind == "nanobeam":
            for name, design in _designs_for_cell(cell, man):
                for res in man.get("res", [20]):
                    p = dict(base, res=int(res), design=design)
                    if not man.get("ensemble"):
                        jobs.append(make_job(exp, "nanobeam3d", p, tag=f"{name}|res{res}|nominal"))
                        continue
                    ens = man["ensemble"]
                    jobs.append(make_job(exp, "nanobeam3d", p, tag=f"{name}|res{res}|nominal"))
                    for cls in ens["classes"]:
                        for sig in ens["sigmas_nm"]:
                            for k in range(int(ens["N"])):
                                dis = dict(seed=int(ens.get("seed", 2026)), realization=k,
                                           sigma_r_nm=0.0, sigma_x_nm=0.0, sigma_y_nm=0.0, sigma_bias_nm=0.0)
                                for c in (cls if isinstance(cls, list) else [cls]):
                                    dis[{"radius": "sigma_r_nm", "position_x": "sigma_x_nm",
                                         "offset_y": "sigma_y_nm", "cd_bias": "sigma_bias_nm"}[c]] = float(sig)
                                jobs.append(make_job(exp, "nanobeam3d", dict(p, disorder=dis),
                                                     tag=f"{name}|res{res}|{'+'.join(cls) if isinstance(cls, list) else cls}|s{sig}|k{k}"))
        elif kind == "ring":
            L = cell["L"]
            for sc in man.get("supercells", []):
                N = int(round(2 * math.pi * float(sc["R_um"]) / L)); N1 = N // 2
                p = dict(base, a=geo["a"], b=geo["b"], r=geo["r"], N1=N1, N2=N - N1, res=int(sc.get("res", 20)))
                jobs.append(make_job(exp, "supercell3d", p, tag=f"supercell|R{sc['R_um']}"))
            for rg in man.get("rings", []):
                N = int(round(2 * math.pi * float(rg["R_um"]) / L)); N1 = N // 2
                p = dict(base, a=geo["a"], b=geo["b"], r=geo["r"], N1=N1, N2=N - N1, res=int(rg.get("res", 20)))
                jobs.append(make_job(exp, "ring3d", p, tag=f"ring|R{rg['R_um']}"))
    return jobs


# --------------------------------------------------------------------------
# Size, routing and cost
# --------------------------------------------------------------------------
def estimate(job: dict) -> dict:
    """Voxels, steps, memory. Mirrors the cell construction in tpc_physics.py."""
    k, p = job["kernel"], job["params"]
    plat = G.platform(p["platform"], t=p.get("t"))
    t, w = plat["t"], p.get("w", 0.5)
    res = int(p.get("res", 20))
    dpml = 1.0
    z_sym = p.get("clad") in ("oxide", "oxide_fill", "susp")
    if k == "mpb_cell":
        return dict(kind="mpb", vox_reduced=0, steps=0)
    if k == "bend_q":
        R = p["R"]; sr = R + w / 2 + 2.5 + dpml; sz = t + 3 + 2 * dpml
        steps = (10 / (0.25 / 1.547) + p["T_after"]) * 2 * res
        return dict(kind="cyl", vox_reduced=sr * sz * res ** 2, steps=steps)
    if k == "nanobeam3d":
        holes = G.build_design(p["design"])
        xmax = max(abs(h["x"]) for h in holes)
        sx = 2 * (xmax + 1.0) + 2 * dpml
        sy = w + 2 * p.get("pad_y", 1.0) + 2 * dpml
        sz = t + 2 * p.get("pad_z", 0.8) + 2 * dpml
        nsym = (0 if G.breaks_y_mirror(p.get("disorder")) else 1) + (1 if z_sym else 0)
    elif k == "supercell3d":
        sx = (p["N1"] + p["N2"]) * (p["a"] + p["b"])
        sy = w + 2 * p.get("pad_y", 1.0) + 2 * dpml
        sz = t + 2 * p.get("pad_z", 0.8) + 2 * dpml
        nsym = 1 + (1 if z_sym else 0)
    elif k == "ring3d":
        R = (p["N1"] + p["N2"]) * (p["a"] + p["b"]) / (2 * math.pi)
        sx = sy = 2 * (R + w / 2 + p.get("pad_r", 1.0)) + 2 * dpml
        sz = t + 2 * p.get("pad_z", 0.8) + 2 * dpml
        nsym = 1 if z_sym else 0
    else:
        raise ValueError(k)
    vox = sx * sy * sz * res ** 3
    vred = vox / 2 ** nsym
    steps = (10.0 / p["df"] + p["T_harm"]) * 2 * res
    return dict(kind="3d", vox_full=vox, vox_reduced=vred, steps=steps, mem_GB=BYTES_PER_REDUCED_VOXEL * vred / 1e9)


def route(job: dict) -> str:
    k = job["kernel"]
    if k in ("mpb_cell", "bend_q"):
        return "light"
    if k == "ring3d":
        return "ring"
    e = estimate(job)
    if k == "supercell3d" or e["vox_reduced"] > 4e6:
        return "3d_big"
    return "3d"


def cost(job: dict, core_ns: float = CORE_NS_PER_VOXEL_STEP) -> dict:
    tier = TIERS[job["tier"]]
    e = estimate(job)
    if e["kind"] == "mpb":
        hours = 60 / 3600  # 5-60 s per MPB solve, a few solves
    elif e["kind"] == "cyl":
        hours = e["vox_reduced"] * e["steps"] * 6e-9 / 3600  # 2D: about 6 ns per pixel-step (pilot BENCH2D)
    else:
        npr = tier["np"]
        hours = e["vox_reduced"] * e["steps"] * core_ns * 1e-9 / (npr * PAR_EFF.get(npr, 0.4)) / 3600
    hours += 0.5 / 60  # amortised container start (map reuses warm containers), rough
    usd = hours * 3600 * (tier["cpu"] * CPU_USD_PER_CORE_S + tier["memory"] / 1024 * MEM_USD_PER_GIB_S)
    return dict(hours=hours, usd=usd, **e)


def summarize_cost(jobs: list[dict]) -> dict:
    tot_h = tot_usd = 0.0; longest = 0.0
    by_tier: dict[str, int] = {}
    for j in jobs:
        c = cost(j); tot_h += c["hours"]; tot_usd += c["usd"]; longest = max(longest, c["hours"])
        by_tier[j["tier"]] = by_tier.get(j["tier"], 0) + 1
    return dict(n_jobs=len(jobs), container_hours=tot_h, usd=tot_usd, longest_job_hours=longest, by_tier=by_tier)


# --------------------------------------------------------------------------
# Ensemble statistics (E4)
# --------------------------------------------------------------------------
def _boot(x: np.ndarray, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    s = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def ensemble_stats(rows: list[dict]) -> dict:
    """rows: nanobeam3d results of one (design, res, class, sigma) cell."""
    q = np.array([r["Q_total"] for r in rows if r.get("Q_total")], float)
    lam = np.array([r["convergence"]["primary"]["lam_um"] for r in rows if r.get("Q_total")], float)
    qr = np.array([r["Q_rad"] for r in rows if r.get("Q_rad")], float)
    if len(q) == 0:
        return dict(n=0)
    hm = lambda x: float(1 / np.mean(1 / x))
    q10 = lambda x: float(np.percentile(x, 10))
    return dict(n=int(len(q)), n_failed=len(rows) - int(len(q)),
                Q_mean=float(q.mean()), Q_median=float(np.median(q)), Q_harmonic=hm(q), Q_harmonic_ci95=_boot(q, hm),
                Q10=q10(q), Q10_ci95=_boot(q, q10), Q_rad_harmonic=hm(qr) if len(qr) else None,
                sigma_lambda_nm=float(1e3 * lam.std(ddof=1)) if len(lam) > 1 else None,
                sigma_lambda_ci95_nm=[1e3 * v for v in _boot(lam, lambda x: x.std(ddof=1))] if len(lam) > 2 else None)


def group_ensemble(results: list[tuple[str, dict]], jobs_by_key: dict[str, dict]) -> dict:
    """Group (key, result) pairs into ensemble cells by tag, dropping the realization index."""
    cells: dict[str, list[dict]] = {}
    for key, r in results:
        parts = [t for t in jobs_by_key[key]["tag"].split("|") if t != "validation"]
        parts = [t for t in parts if not (t.startswith("k") and t[1:].isdigit())]
        cells.setdefault("|".join(parts), []).append(r)
    out = {}
    for cell, rows in sorted(cells.items()):
        if cell.endswith("|nominal"):
            r = rows[0]
            out[cell] = dict(status=r.get("status"), Q_total=r.get("Q_total"), Q_rad=r.get("Q_rad"),
                             lam_um=((r.get("convergence") or {}).get("primary") or {}).get("lam_um"))
        else:
            out[cell] = ensemble_stats(rows)
    return out


# --------------------------------------------------------------------------
# Go / no-go checks (criteria are stated in docs/MODAL_RUNBOOK.md)
# --------------------------------------------------------------------------
def _ok(rows):
    return [r for r in rows if r.get("status", "ok") == "ok"]


def gonogo(exp: str, rows: list[dict]) -> dict:
    rows = [r for r in rows if isinstance(r, dict)]
    if exp == "E0":
        mpb = [r for r in rows if r.get("kernel") == "mpb_cell"]
        cyl = [r for r in rows if r.get("kernel") == "bend_q"]
        go = bool(mpb) and bool(cyl) and all(r.get("gap_abs", 0) > 0 for r in mpb) and all(r.get("Q_bend") for r in cyl)
        return dict(exp=exp, go=go, mpb=[{k: r.get(k) for k in ("lambda1_um", "lambda2_um", "dlambda_nm", "wall_s")} for r in mpb],
                    bend=[{k: r.get(k) for k in ("R", "w", "m", "Q_bend", "lam_bend_um", "kernel_wall_s")} for r in cyl],
                    criterion="both kernels ran on Modal, MPB gap > 0, Harminv found a bend mode")
    if exp == "E1":
        tab = sorted([(r["platform"], r["clad"], r["w"], r["R"], r.get("Q_bend"), r.get("lam_bend_um"),
                       (r.get("convergence") or {}).get("dQ_rel_half_vs_full")) for r in rows if r.get("kernel") == "bend_q"],
                     key=lambda x: (x[0], x[1], x[2], x[3]))
        best = {}
        for p_, c, w, R, q, lam, dq in tab:
            if q and q >= 1e5 and (p_ not in best or R < best[p_]["R"]):
                best[p_] = dict(clad=c, w=w, R=R, Q_bend=q, dQ_half=dq)
        return dict(exp=exp, go=bool(best), smallest_R_with_Qbend_ge_1e5=best,
                    table=[dict(platform=a, clad=b, w=c, R=d, Q_bend=e, lam_um=f, dQ_half=g) for a, b, c, d, e, f, g in tab],
                    criterion="some (platform, w, clad) reaches Q_bend >= 1e5 at a usable R; Q at T/2 and T agree within 10%")
    if exp == "E2":
        cells = [r for r in rows if r.get("kernel") == "mpb_cell"]
        by = {}
        for r in cells:
            k = f"{r['platform']}|{r['clad']}|w{r['w']}"
            b = by.setdefault(k, dict(n=0, fab_ok=0, feasible=0, best=None))
            b["n"] += 1
            b["fab_ok"] += int(bool(r.get("fab", {}).get("ok")))
            feas = r.get("fab", {}).get("ok") and r.get("gap_contains_target") and r.get("below_lightline")
            if feas:
                b["feasible"] += 1
                if not b["best"] or r["gap_mid_ratio"] > b["best"]["gap_mid_ratio"]:
                    b["best"] = {k2: r[k2] for k2 in ("a", "b", "r", "L", "dlambda_nm", "gap_mid_ratio", "lambda_mid_um")}
                    b["best"]["min_web_nm"] = r["fab"]["min_web_nm"]
        refused = {}
        for r in cells:
            for v in r.get("fab", {}).get("violations", []):
                refused[v.split(" ")[0] + " " + v.split(" ")[1]] = refused.get(v.split(" ")[0] + " " + v.split(" ")[1], 0) + 1
        return dict(exp=exp, go=any(b["feasible"] for b in by.values()), by_stack=by, violation_counts=refused,
                    criterion="at least one fabricable cell (rules of its platform) whose zone-edge gap contains 1547 nm, below the light line")
    if exp in ("E3", "E5"):
        ok = _ok(rows)
        ref = [dict(tag=r.get("job", {}).get("tag"), reason=r.get("reason")) for r in rows if r.get("status") == "refused_fab"]
        if exp == "E3":
            pairs = {}
            for r in ok:
                tag = r.get("job", {}).get("tag", "")
                name = tag.split("|")[0]
                pm = (r.get("convergence") or {}).get("primary")
                pairs.setdefault(name, {})[r["res"]] = dict(Q=r.get("Q_total"), Q_rad=r.get("Q_rad"), Q_axial=r.get("Q_axial"),
                                                            lam_um=pm["lam_um"] if pm else None, T_harm=r.get("T_harm"),
                                                            dQ_half=(r.get("convergence") or {}).get("dQ_rel_half_vs_full"))
            conv = {}
            for name, d in pairs.items():
                rs = sorted(d)
                if len(rs) >= 2 and d[rs[0]]["Q"] and d[rs[-1]]["Q"]:
                    q0, q1 = d[rs[0]]["Q"], d[rs[-1]]["Q"]
                    conv[name] = dict(dQ_rel=abs(q1 - q0) / q1, dlam_nm=1e3 * abs(d[rs[-1]]["lam_um"] - d[rs[0]]["lam_um"]))
            go = bool(conv) and all(c["dQ_rel"] < 0.2 and c["dlam_nm"] < 2.0 for c in conv.values())
            return dict(exp=exp, go=go, results=pairs, convergence=conv, refused=ref,
                        criterion="every design has a mode in the window; res 20 vs 30: |dQ|/Q < 20% and |dlambda| < 2 nm; "
                                  "Quan-Loncar control has a mode in the window")
        return dict(exp=exp, go=bool(ok), refused=ref,
                    results=[dict(tag=r.get("job", {}).get("tag"), primary_A=r.get("primary_A"), primary_B=r.get("primary_B"),
                                  primary=(r.get("convergence") or {}).get("primary")) for r in ok],
                    criterion="supercell shows separate A and B wall modes; full ring Q within ~3x of the Q budget")
    return dict(exp=exp, go=None, n=len(rows))


if __name__ == "__main__":  # offline plan + cost: python research/modal/tpc_plan.py E1 [E2 ...]
    import sys
    for e in sys.argv[1:] or ["E0", "E1", "E2"]:
        man = load_manifest(e)
        if man.get("select"):
            print(f"{e}: selects geometry from {man['select']['from']} results; plan it with "
                  f"`modal run research/modal/app.py::{e.lower()} --dry-run` once those exist")
            continue
        print(e, json.dumps(summarize_cost(expand(man))))
