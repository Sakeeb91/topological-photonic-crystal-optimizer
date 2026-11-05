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
    "3d_xl":  dict(cpu=32.0, memory=32768, timeout=24 * 3600, np=32),  # res >= 60 nanobeams
    "ring":   dict(cpu=32.0, memory=32768, timeout=24 * 3600, np=32),  # 24 h is Modal's maximum
}

# MEEP 1.35 MPICH cost in core-ns per symmetry-reduced voxel-step, MEASURED ON MODAL (first E3 run,
# 2026-11-05, 16 nanobeam3d jobs, field ns_core_per_reduced_voxel_step; research/results/):
#   np=8: median 26 (range 19-28);  np=16: median 40 (range 24-69).
# np=1-4 and np=32 are ASSUMED (unmeasured on Modal): the 8-rank value below 8, and 40/0.7 for 32.
# (The 4-core cloud container measured 63 core-ns at np=2 and 92 at np=4: Modal cores are ~3x faster.)
CORE_NS_BY_NP = {1: 26.0, 2: 26.0, 4: 26.0, 8: 26.0, 16: 40.0, 32: 57.0}
CORE_NS_PER_VOXEL_STEP = CORE_NS_BY_NP[8]  # scale factor for callers that pass core_ns explicitly
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
    if sel.get("min_r_nm"):  # larger holes need lower resolution: cost scales as res^4 ~ r^-4
        rows = [r for r in rows if 1e3 * r["r"] >= float(sel["min_r_nm"]) - 1e-9]
    margin = float(sel.get("min_margin_nm", 0.0))
    if margin > 0:  # extra web/feature margin above the platform rule, against CD bias (tapeout safety)
        def _m(r):
            pl = G.platform(r["platform"])
            return min(r["fab"]["min_web_nm"] - 1e3 * pl["min_web"], r["fab"]["min_hole_diameter_nm"] - 1e3 * pl["min_feature"])
        rows = [r for r in rows if _m(r) >= margin - 1e-9]
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
    sc = float(ql.get("period_scale", 1.0))
    if sc != 1.0:  # retune the control's resonance onto the target; radii scale with the period
        L, r_c, r_end = L * sc, r_c * sc, r_end * sc
    return dict(kind="quan_loncar", period=L, r_center=r_c, r_end=r_end,
                n_taper=int(ql.get("n_taper", 10)), n_mirror=int(ql.get("n_mirror", 10)))


def _variants(d: dict) -> list[tuple[dict, str]]:
    """Expand list-valued fields of a manifest design entry (e.g. Nside: [20, 30, 40]) into variants.
    Returns (entry, suffix); the suffix names the swept values so variants stay distinguishable."""
    keys = [k for k, v in d.items() if isinstance(v, list)]
    if not keys:
        return [(d, "")]
    out = []
    for combo in itertools.product(*[d[k] for k in keys]):
        dd = dict(d); dd.update(zip(keys, combo))
        out.append((dd, "[" + ",".join(f"{k}={v}" for k, v in zip(keys, combo)) + "]"))
    return out


def _designs_for_cell(cell: dict, man: dict) -> list[tuple[str, dict]]:
    plat = G.platform(cell["platform"])
    out = []
    for d0 in man.get("designs", []):
        for d, sfx in _variants(d0):
            if d["kind"] == "ssh":
                dd = dict(kind="ssh", a=cell["a"], b=cell["b"], r=cell["r"], Nside=int(d.get("Nside", 20)), wall=d["wall"])
                if d["wall"] == "tanh":
                    dd["M"] = float(d.get("M", 6))
                out.append((d.get("name", f"ssh_{d['wall']}") + sfx, dd))
            elif d["kind"] == "quan_loncar":
                out.append((d.get("name", "quan_loncar") + sfx, _ql_design(cell, d, plat)))
    return out


def resolutions(man: dict, design: dict) -> list[int]:
    """Resolution list for one design. `px_on_r: [1.5, 2, 2.5]` sets res = ceil(px / r_min) so the
    smallest hole radius gets that many pixels (the criterion that matters for small Si holes);
    otherwise the manifest's fixed `res` list."""
    if man.get("px_on_r"):
        rmin = min(h["r"] for h in G.build_design(design))
        return sorted({int(math.ceil(float(px) / rmin)) for px in man["px_on_r"]})
    return [int(x) for x in man.get("res", [20])]


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
        if man.get("check_other_parity", True):
            # TE1/TM-like bands inside the TE0 SSH gap would couple under transverse disorder
            # (likely for multimode cross-sections such as SiN w >= 1.0 um at 400 nm).
            jobs.append(make_job(exp, "mpb_cell", dict(platform=geo["platform"], clad=geo["clad"], w=geo["w"],
                                                       a=geo["a"], b=geo["b"], r=geo["r"], res=int(cell.get("res", 32)),
                                                       Ly=cell.get("Ly", 3.0), Lz=cell.get("Lz", 3.0),
                                                       check_other_parity=True), tag="parity_check"))
        base = dict(platform=geo["platform"], w=geo["w"], clad=geo["clad"], fcen=fcen, df=df, **sim)
        kind = man.get("kind", "nanobeam")
        if kind == "nanobeam":
            for name, design in _designs_for_cell(cell, man):
                for res in resolutions(man, design):
                    p = dict(base, res=int(res), design=design)
                    if not man.get("ensemble"):
                        jobs.append(make_job(exp, "nanobeam3d", p, tag=f"{name}|res{res}|nominal"))
                        continue
                    ens = man["ensemble"]
                    jobs.append(make_job(exp, "nanobeam3d", p, tag=f"{name}|res{res}|nominal"))
                    if man.get("pt"):
                        sp = dict(p, **{k: v for k, v in man["pt"].items() if k in ("n_theta", "n_z", "T_harm_pass1")})
                        jobs.append(make_job(exp, "sens3d", sp, tag=f"{name}|res{res}|sens"))
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
# Budget (program cap set by the user: $10 of Modal spend in total, docs/EXPERIMENT_LOG.md)
# --------------------------------------------------------------------------
BUDGET_USD = 10.0
PROGRAM_APPS = ("tpc-experiments", "ssh-ring-smoke")  # apps whose spend counts against the cap
PROGRAM_START = "2026-11-05"
BUDGET_SAFETY = 1.6  # measured/model wall time reached 1.59 on the first E3 run (median 0.82)


def program_spend(items, apps=PROGRAM_APPS) -> float:
    """Sum billing-report rows (dicts or objects with description/cost) for the program's apps."""
    tot = 0.0
    for it in items:
        desc = it["description"] if isinstance(it, dict) else getattr(it, "description", "")
        cost = it["cost"] if isinstance(it, dict) else getattr(it, "cost", 0)
        if desc in apps:
            tot += float(cost)
    return tot


def budget_check(spent_usd: float, est_usd: float, cap_usd: float = BUDGET_USD, safety: float = BUDGET_SAFETY) -> dict:
    """Refuse when already-spent + safety x estimate would exceed the cap."""
    worst = spent_usd + safety * est_usd
    return dict(ok=worst <= cap_usd, spent_usd=spent_usd, est_usd=est_usd, worst_case_usd=worst,
                cap_usd=cap_usd, headroom_usd=cap_usd - spent_usd)


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
    if k in ("nanobeam3d", "sens3d"):
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
    mem = BYTES_PER_REDUCED_VOXEL * vred / 1e9
    if k == "sens3d":  # two passes; DFT of 6 complex components over the slab box, gathered in full
        steps *= 2
        dft_pts = (sx - 2 * dpml) * (w + 1.2) * (t + 1.0) * res ** 3
        # get_dft_array returns the full array on every MPI rank: 6 complex components x (array +
        # interpolator copy), times 16 ranks (the 3d_big tier) as a conservative bound
        mem += 6 * 16 * dft_pts * 2 * 16 / 1e9
    return dict(kind="3d", vox_full=vox, vox_reduced=vred, steps=steps, mem_GB=mem)


def route(job: dict) -> str:
    k = job["kernel"]
    if k in ("mpb_cell", "bend_q"):
        return "light"
    if k == "ring3d":
        return "ring"
    e = estimate(job)
    if e["vox_reduced"] > 1.5e7 or e.get("mem_GB", 0) > 0.8 * TIERS["3d_big"]["memory"] / 1024:
        return "3d_xl"
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
        cns = CORE_NS_BY_NP.get(npr, 60.0) * core_ns / CORE_NS_PER_VOXEL_STEP
        hours = e["vox_reduced"] * e["steps"] * cns * 1e-9 / npr / 3600
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


_CLASS_KEY = {"radius": "sigma_r_nm", "position_x": "sigma_x_nm", "offset_y": "sigma_y_nm", "cd_bias": "sigma_bias_nm"}


def pt_shift(sens: dict, design: dict, dis: dict) -> float:
    """First-order frequency shift (1/um) for one CRN disorder realization, from sens3d gradients."""
    g = {h["ord"]: h for h in sens["holes"]}
    nom = G.build_design(design)
    per = G.apply_disorder(nom, dis)
    return float(sum(g[a["ord"]]["df_dr"] * (b["r"] - a["r"]) + g[a["ord"]]["df_dx"] * (b["x"] - a["x"])
                     + g[a["ord"]]["df_dy"] * (b["y"] - a["y"]) for a, b in zip(nom, per)))


def pt_stats(sens: dict, design: dict, classes, sigmas_nm, n_draws: int = 2000, seed: int = 2026) -> dict:
    """sigma_lambda per (class, sigma) from first-order PT. Linear in the displacements, so the
    analytic value is exact for the linear model; the CRN Monte Carlo is reported alongside.
    Draws use realization indices 100000+ so they never coincide with FDTD spot checks."""
    lam0 = sens["lam0_um"]
    gr = np.array([h["df_dr"] for h in sens["holes"]]); gx = np.array([h["df_dx"] for h in sens["holes"]])
    gy = np.array([h["df_dy"] for h in sens["holes"]])
    analytic_coef = {"radius": float(np.sqrt((gr ** 2).sum())), "position_x": float(np.sqrt((gx ** 2).sum())),
                     "offset_y": float(np.sqrt((gy ** 2).sum())), "cd_bias": float(abs(gr.sum()))}
    out = {}
    for cls in classes:
        cl = cls if isinstance(cls, list) else [cls]
        for sig in sigmas_nm:
            an = 1e-3 * float(sig) * float(np.sqrt(sum(analytic_coef[c] ** 2 for c in cl)))
            dis = {v: 0.0 for v in _CLASS_KEY.values()}
            for c in cl:
                dis[_CLASS_KEY[c]] = float(sig)
            df = np.array([pt_shift(sens, design, dict(dis, seed=seed, realization=100000 + k)) for k in range(n_draws)])
            out[f"{'+'.join(cl)}|s{sig}"] = dict(sigma_lambda_nm_analytic=1e3 * lam0 ** 2 * an,
                                                sigma_lambda_nm_mc=float(1e3 * lam0 ** 2 * df.std(ddof=1)), n_draws=n_draws)
    return dict(lam0_um=lam0, dlam_dbias_nm_per_nm=sens.get("dlam_dbias_nm_per_nm"), by_class=out,
                note="first-order frequency statistics; Q under disorder needs FDTD (loss is second order)")


def group_ensemble(results: list[tuple[str, dict]], jobs_by_key: dict[str, dict], pt: dict | None = None) -> dict:
    """Per (design, res) ensemble cell: FDTD statistics; with sens3d results, PT sigma_lambda for all
    classes/sigmas, PT-vs-FDTD spot-check agreement, and the Minkov coefficient C in
    <1/Q - 1/Q_nominal> = C sigma^2 (fit per class from the FDTD spot set)."""
    cells: dict[str, list] = {}
    sens: dict[str, dict] = {}
    nominal: dict[str, dict] = {}
    for key, r in results:
        job = jobs_by_key[key]
        parts = [t for t in job["tag"].split("|") if t != "validation"]
        if job["kernel"] not in ("nanobeam3d", "sens3d") or len(parts) < 3:
            continue  # e.g. the parity_check MPB job
        base = "|".join(parts[:2])
        if parts[2] == "sens":
            sens[base] = r; continue
        if parts[2] == "nominal":
            nominal[base] = r; continue
        cell = "|".join(t for t in parts if not (t.startswith("k") and t[1:].isdigit()))
        cells.setdefault(cell, []).append((job, r))
    out = {}
    for base, r in sorted(nominal.items()):
        out[f"{base}|nominal"] = dict(status=r.get("status"), Q_total=r.get("Q_total"), Q_rad=r.get("Q_rad"),
                                      lam_um=((r.get("convergence") or {}).get("primary") or {}).get("lam_um"))
    for cell, pairs in sorted(cells.items()):
        base = "|".join(cell.split("|")[:2])
        rows = [r for _, r in pairs]
        st = ensemble_stats(rows)
        nr = nominal.get(base)
        sig = float(cell.split("|")[-1][1:])
        if nr and nr.get("Q_total"):
            inv = [1 / r["Q_total"] - 1 / nr["Q_total"] for r in rows if r.get("Q_total")]
            if inv:
                st["C_minkov_per_nm2"] = float(np.mean(inv) / sig ** 2)
                st["C_note"] = "<1/Q - 1/Q_nom> / sigma^2 (Minkov et al., Opt. Express 21, 28233, 2013); extrapolates loss to 1-3 nm"
        if base in sens and sens[base].get("status") == "ok" and nr:
            f_nom = ((nr.get("convergence") or {}).get("primary") or {}).get("f")
            cmp = []
            for job, r in pairs:
                pm = (r.get("convergence") or {}).get("primary") or {}
                if f_nom and pm.get("f"):
                    cmp.append((pm["f"] - f_nom, pt_shift(sens[base], job["params"]["design"], job["params"]["disorder"])))
            if cmp:
                a = np.array(cmp)
                st["pt_check"] = dict(n=len(a), fdtd_df=a[:, 0].tolist(), pt_df=a[:, 1].tolist(),
                                      slope_fdtd_vs_pt=float((a[:, 0] @ a[:, 1]) / max(a[:, 1] @ a[:, 1], 1e-30)),
                                      corr=float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else None)
        out[cell] = st
    for base, sr in sorted(sens.items()):
        if sr.get("status") == "ok" and pt:
            out[f"{base}|pt"] = pt_stats(sr, sr["design"], pt.get("classes", ["radius", "position_x", "offset_y", "cd_bias"]),
                                         pt.get("sigmas_nm", [1, 2, 3, 5, 10]), int(pt.get("n_draws", 2000)), int(pt.get("seed", 2026)))
        else:
            out[f"{base}|pt"] = dict(status=sr.get("status"), reason=sr.get("reason"))
    return out


# --------------------------------------------------------------------------
# Go / no-go checks (criteria are stated in docs/MODAL_RUNBOOK.md)
# --------------------------------------------------------------------------
def _ok(rows):
    return [r for r in rows if r.get("status", "ok") == "ok"]


Q_CAP = 1e6          # Harminv/FDTD cannot resolve bend Q above ~1e6-1e7 at the E1 run lengths and PML
Q_DECISION = 1e5     # E1 decides which (platform, w, R) reach Q_bend >= 1e5


def _q_status(q, dq_half, cap=Q_CAP):
    """'ok' (resolved), 'above_cap' (resolved and > cap), 'unresolved' (Q <= 0 or T/2-vs-T change > 10%)."""
    if q is None or q <= 0 or (dq_half is not None and dq_half > 0.1):
        return "unresolved"
    return "above_cap" if q > cap else "ok"


def _gonogo_e1(rows: list[dict], cap: float = Q_CAP, thr: float = Q_DECISION) -> dict:
    bq = [r for r in rows if r.get("kernel") == "bend_q"]
    tab = []
    for r in bq:
        dq = (r.get("convergence") or {}).get("dQ_rel_half_vs_full")
        tab.append(dict(platform=r["platform"], clad=r["clad"], w=r["w"], R=r["R"], res=r["res"],
                        T_after=r.get("T_harm", r.get("T_after")), Q_bend=r.get("Q_bend"), lam_um=r.get("lam_bend_um"),
                        dQ_half=dq, q_status=_q_status(r.get("Q_bend"), dq, cap)))
    tab.sort(key=lambda d: (d["platform"], d["clad"], d["w"], d["R"], d["res"]))
    groups: dict[tuple, list[dict]] = {}
    for d in tab:
        groups.setdefault((d["platform"], d["clad"], d["w"], d["R"]), []).append(d)
    conv = {}
    for k, g in groups.items():
        if len(g) < 2:
            continue
        lo, hi = g[0], g[-1]  # lowest and highest resolution
        st = {lo["q_status"], hi["q_status"]}
        e = dict(res=[lo["res"], hi["res"]], T=[lo["T_after"], hi["T_after"]], Q=[lo["Q_bend"], hi["Q_bend"]])
        any_high = any((d["Q_bend"] or 0) > cap for d in (lo, hi))  # positive and above the cap, resolved or not
        if st <= {"above_cap", "unresolved"} and any_high and "ok" not in st:
            e.update(verdict="agree", Q_display=f">{cap:.0e}",
                     note="both at or beyond the resolvable limit: bend loss negligible" if "unresolved" not in st
                     else f"one member unresolved, the other > {cap:.0e}: treated as > cap")
        elif "unresolved" in st:
            e.update(verdict="flag", note="unresolved member without an above-cap partner: rerun with longer T_after")
        else:
            q0, q1 = min(lo["Q_bend"], cap), min(hi["Q_bend"], cap)
            e["dQ_rel"] = abs(q1 - q0) / q1
            e["dlam_nm"] = 1e3 * abs(hi["lam_um"] - lo["lam_um"])
            if e["dQ_rel"] < 0.2:
                e["verdict"] = "agree"
            elif (lo["Q_bend"] >= thr) == (hi["Q_bend"] >= thr):
                e.update(verdict="same_side_of_threshold",
                         note=f"|dQ|/Q = {e['dQ_rel']:.0%} but both on the same side of {thr:.0e}: the E1 decision is unchanged")
            else:
                e.update(verdict="flag", note=f"members straddle the {thr:.0e} decision threshold")
        conv[f"{k[0]}|{k[1]}|w{k[2]}|R{k[3]}"] = e
    flagged = [k for k, e in conv.items() if e["verdict"] == "flag"]
    # smallest R per platform whose (highest-res) row is resolved and >= thr, or above cap
    best = {}
    for k, g in groups.items():
        d = g[-1]
        if d["q_status"] == "unresolved" and not any((x["Q_bend"] or 0) > cap for x in g):
            continue
        qok = any((x["Q_bend"] or 0) > cap for x in g) or (d["q_status"] == "ok" and (d["Q_bend"] or 0) >= thr)
        if qok and (d["platform"] not in best or (d["R"], d["w"]) < (best[d["platform"]]["R"], best[d["platform"]]["w"])):
            best[d["platform"]] = dict(clad=d["clad"], w=d["w"], R=d["R"], res=d["res"],
                                       Q_bend=(f">{cap:.0e}" if (d["Q_bend"] or 0) > cap or d["q_status"] != "ok" else d["Q_bend"]))
    return dict(exp="E1", go=bool(best) and bool(conv) and not flagged, smallest_R_with_Qbend_ge_1e5=best,
                resolution_convergence=conv, flagged_pairs=flagged, Q_cap=cap, table=tab,
                criterion=f"some (platform, w, clad) reaches Q_bend >= {thr:.0e}; every convergence pair agrees "
                          f"(|dQ|/Q < 0.2 below Q_cap = {cap:.0e}, or both >= cap) or stays on the same side of {thr:.0e}; "
                          "unresolved rows (Q <= 0 or T/2-vs-T change > 10%) agree as '>cap' when no member is a resolved below-cap "
                          "value and at least one member is positive and > cap")


def _cell_key(design: dict) -> str:
    L = design["a"] + design["b"] if design["kind"] == "ssh" else design.get("L", design.get("period"))
    return f"L{1e3 * L:.1f}nm"


def _gonogo_e3(rows, ok, ref, target_um: float = 1.547, mirror_ratio: float = 10.0) -> dict:
    """Group by (cell, design variant incl. Nside), compare Q_rad (not Q_total) and lambda across the
    two highest resolutions, and flag mirror-limited variants (Q_axial < mirror_ratio * Q_rad)."""
    groups: dict[str, dict] = {}
    for r in [x for x in ok if x.get("kernel") == "nanobeam3d"]:
        d = r["design"]
        name = r.get("job", {}).get("tag", "").split("|")[0]
        var = {k: v for k, v in d.items() if k not in ("a", "b", "r", "period", "r_center", "r_end")}
        key = f"{_cell_key(d)}|{name}|" + ",".join(f"{k}={v}" for k, v in sorted(var.items()) if k != "kind")
        pm = (r.get("convergence") or {}).get("primary") or {}
        g = groups.setdefault(key, dict(cell=_cell_key(d), design=name.split("[")[0], variant=name, by_res={}))
        rmin = min(h["r"] for h in G.build_design(d))
        g["by_res"][r["res"]] = dict(Q_total=r.get("Q_total"), Q_rad=r.get("Q_rad"), Q_axial=r.get("Q_axial"),
                                     lam_um=pm.get("lam_um"), px_on_rmin=rmin * r["res"], T_harm=r.get("T_harm"),
                                     window_um=[1 / (r["fcen"] + r["df"] / 2), 1 / (r["fcen"] - r["df"] / 2)])
    for key, g in groups.items():
        rs = sorted(g["by_res"])
        top = g["by_res"][rs[-1]]
        g["max_res"] = rs[-1]
        g["n_res"] = len(rs)
        g["mirror_limited"] = bool(top["Q_axial"] and top["Q_rad"] and top["Q_axial"] < mirror_ratio * top["Q_rad"])
        lam = top["lam_um"]
        g["in_window"] = bool(lam and top["window_um"][0] <= lam <= top["window_um"][1])
        g["detuning_nm"] = 1e3 * (lam - target_um) if lam else None
        if len(rs) >= 2:
            p, q = g["by_res"][rs[-2]], top
            if p["Q_rad"] and q["Q_rad"] and p["lam_um"] and q["lam_um"]:
                g["dQrad_rel"] = abs(q["Q_rad"] - p["Q_rad"]) / q["Q_rad"]
                g["dlam_nm"] = 1e3 * abs(q["lam_um"] - p["lam_um"])
        g["converged"] = bool(g["n_res"] >= 3 and g.get("dQrad_rel", 9) < 0.2 and g.get("dlam_nm", 99) < 2.0
                              and top["px_on_rmin"] >= 2.0)
        g["usable"] = g["converged"] and not g["mirror_limited"] and g["in_window"]
    designs = sorted({(g["cell"], g["design"]) for g in groups.values()})
    status = {f"{c}|{d}": any(g["usable"] for g in groups.values() if g["cell"] == c and g["design"] == d) for c, d in designs}
    par = [dict(a=r["a"], b=r["b"], r=r["r"], w=r["w"], clear=r.get("other_parity_clear"),
                in_gap=r.get("other_parity_in_gap")) for r in rows if r.get("kernel") == "mpb_cell"]
    has_ql = any(d == "quan_loncar" for _, d in designs)
    return dict(exp="E3", go=bool(status) and all(status.values()) and has_ql, design_usable=status, variants=groups,
                refused=ref, other_parity=par,
                criterion=f"for every (cell, design) incl. the Quan-Loncar control, some variant is: converged over >= 3 "
                          f"resolutions (top two: |dQ_rad|/Q_rad < 0.2, |dlambda| < 2 nm, >= 2 px on the smallest hole radius), "
                          f"not mirror-limited (Q_axial >= {mirror_ratio:.0f} x Q_rad), and resonant inside the source window. "
                          "Compare designs on Q_rad, not Q_total.")


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
        return _gonogo_e1(rows)
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
                    pl = G.platform(r["platform"])
                    b["best"]["fab_margin_nm"] = min(r["fab"]["min_web_nm"] - 1e3 * pl["min_web"],
                                                     r["fab"]["min_hole_diameter_nm"] - 1e3 * pl["min_feature"])
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
            return _gonogo_e3(rows, ok, ref)
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
