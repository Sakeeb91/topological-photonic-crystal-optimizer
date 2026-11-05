"""
Physics kernels for the Modal experiments. Each kernel takes a params dict and returns a
JSON-serialisable result dict. Run one kernel (under MPI for the 3D ones) with

    [mpirun -np N] python tpc_physics.py <kernel> params.json out.json

Kernels (derived from research/pilot_sims/, same conventions):
    mpb_cell     3D MPB zone-edge SSH gap of one unit cell; optional rescale of the
                 in-plane lattice (a, b, r at fixed w, t) so the gap contains a target lambda
    bend_q       exact 3D bend-loss Q of a bare ring, MEEP cylindrical coordinates;
                 picks m from an MPB n_eff estimate when m is not given
    nanobeam3d   3D straight nanobeam (SSH wall none/A/B/tanh, or Quan-Loncar trivial),
                 optional CRN disorder, Harminv at T/2 and T, flux-box Q split
    supercell3d  unrolled ring: one A and one B wall, periodic in x, length = circumference
    ring3d       full 3D domain-wall ring

Nothing here imports `modal`.
"""
from __future__ import annotations

import json
import math
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpc_geometry as G  # noqa: E402

LAM0 = 1.547  # target wavelength, um (MEEP/MPB frequency unit is 1/um)


def _meep():
    import meep as mp
    mp.verbosity(0)
    return mp


def _materials(mp, plat, clad):
    core = mp.Medium(index=plat["n_core"])
    ox = mp.Medium(index=plat["n_ox"])
    if clad in ("oxide", "oxide_fill"):
        bg = ox
    elif clad in ("sub", "susp"):
        bg = mp.air
    else:
        raise ValueError(f"clad must be one of {G.CLADS}")
    hole = ox if clad == "oxide_fill" else mp.air
    return core, ox, bg, hole


def _z_symmetric(clad):
    return clad in ("oxide", "oxide_fill", "susp")


# ==========================================================================
# MPB unit cell
# ==========================================================================
def _mpb_cell_once(a, b, r, w, t, plat, clad, res, Ly, Lz, num_bands=4, other_parity=False):
    mp = _meep()
    from meep import mpb
    core, ox, bg, hole = _materials(mp, plat, clad)
    L = a + b
    geo = []
    if clad == "sub":
        geo.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - Lz / 4), size=mp.Vector3(mp.inf, mp.inf, Lz / 2), material=ox))
    geo.append(mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t), material=core))
    if r > 0:
        for x0 in (-a / 2, a / 2):
            geo.append(mp.Cylinder(radius=r, height=t, center=mp.Vector3(x0, 0, 0), material=hole))
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(L, Ly, Lz)), geometry=geo,
                        default_material=bg, k_points=[mp.Vector3(0.5)], resolution=int(res),
                        num_bands=num_bands, tolerance=1e-7)
    t0 = time.time()
    ms.run_yodd()  # TE-like (Ey even about y=0) bands, as in the pilot
    wall_s = time.time() - t0
    f = [float(v) for v in ms.all_freqs[-1]]
    n_lc = 1.0 if clad == "susp" else plat["n_ox"]
    f_ll = 0.5 / L / n_lc
    f1, f2 = f[0], f[1]
    mid = 0.5 * (f1 + f2)
    extra = {}
    if other_parity:
        # Opposite y-parity bands (TE1-like and TM-like) at the zone edge. They do not couple to
        # the TE0 SSH bands in a y-symmetric beam, but transverse disorder (offset_y) mixes them.
        ms.run_yeven()
        fo = [float(v) for v in ms.all_freqs[-1]]
        inside = [v for v in fo if f1 < v < f2]
        extra = dict(freqs_edge_other_parity=fo, other_parity_in_gap=inside, other_parity_clear=not inside)
    return dict(a=a, b=b, r=r, L=L, freqs_edge=f, **extra, edge_f1=f1, edge_f2=f2, gap_abs=f2 - f1,
                gap_mid_ratio=(f2 - f1) / mid, lambda1_um=1 / f1, lambda2_um=1 / f2, lambda_mid_um=1 / mid,
                dlambda_nm=1e3 * (1 / f1 - 1 / f2), f_lightline=f_ll, below_lightline=bool(f2 < f_ll),
                wall_s=wall_s)


def mpb_cell(p):
    plat = G.platform(p["platform"], t=p.get("t"))
    w, t, clad, res = p["w"], plat["t"], p["clad"], int(p["res"])
    Ly, Lz = p.get("Ly", 3.0), p.get("Lz", 3.0)
    if "a" in p:
        a, b, r = p["a"], p["b"], p["r"]
    else:  # dimensionless spec: delta = (a-b)/(a+b), rL = r/L, initial period L0
        L0, d, rL = p["L0"], p["delta"], p["r_over_L"]
        a, b, r = L0 * (1 + d) / 2, L0 * (1 - d) / 2, rL * L0
    target = p.get("target_lambda_um")
    hist = []
    res_d = _mpb_cell_once(a, b, r, w, t, plat, clad, res, Ly, Lz)
    hist.append(res_d)
    for _ in range(int(p.get("rescale_iters", 0)) if target else 0):
        if res_d["lambda2_um"] <= target <= res_d["lambda1_um"] and abs(res_d["lambda_mid_um"] - target) < 0.25 * 1e-3 * max(res_d["dlambda_nm"], 1e-3):
            break
        s = target / res_d["lambda_mid_um"]
        a, b, r = a * s, b * s, r * s
        res_d = _mpb_cell_once(a, b, r, w, t, plat, clad, res, Ly, Lz)
        hist.append(res_d)
    out = dict(res_d)
    out.update(platform=plat["name"], w=w, t=t, clad=clad, res=res, Ly=Ly, Lz=Lz,
               target_lambda_um=target, n_mpb_solves=len(hist),
               history=[{k: h[k] for k in ("L", "lambda_mid_um", "dlambda_nm")} for h in hist])
    if target:
        out["gap_contains_target"] = bool(res_d["lambda2_um"] <= target <= res_d["lambda1_um"])
    out["fab"] = G.ssh_cell_fab_check(a, b, r, w, plat)
    if p.get("check_other_parity"):
        op = _mpb_cell_once(a, b, r, w, t, plat, clad, res, Ly, Lz, other_parity=True)
        out.update({k: op[k] for k in ("freqs_edge_other_parity", "other_parity_in_gap", "other_parity_clear")})
    return out


def mpb_neff(plat, w, t, clad, lam=LAM0, res=40, Ly=4.0, Lz=4.0):
    """TE0 effective index of a bare strip (a1_mpb_waveguide_neff.py)."""
    mp = _meep()
    from meep import mpb
    core, ox, bg, _ = _materials(mp, plat, "oxide" if clad == "oxide_fill" else clad)
    geo = []
    if clad == "sub":
        geo.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - Lz / 4), size=mp.Vector3(mp.inf, mp.inf, Lz / 2), material=ox))
    geo.append(mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t), material=core))
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0, Ly, Lz)), geometry=geo,
                        default_material=bg, resolution=res, num_bands=1)
    f0 = 1 / lam
    k = ms.find_k(mp.ODD_Y, f0, 1, 1, mp.Vector3(1, 0, 0), 1e-4, f0 * plat["n_core"] * 0.8, f0 * 1.0, f0 * plat["n_core"])
    return float(k[0] / f0)


# ==========================================================================
# Harminv helpers
# ==========================================================================
def _modes(hs, fmin, fmax, nmax=10):
    seen = {}
    for h in hs:
        for m in h.modes:
            f = float(m.freq)
            if not (fmin <= f <= fmax):
                continue
            key = round(f, 6)
            d = dict(f=f, lam_um=1 / f, Q=float(m.Q), amp=float(abs(m.amp)), err=float(abs(m.err)))
            if key not in seen or d["amp"] > seen[key]["amp"]:
                seen[key] = d
    return sorted(seen.values(), key=lambda d: -d["amp"])[:nmax]


def _primary(modes, err_max=1e-3):
    good = [m for m in modes if m["err"] < err_max and m["Q"] > 0]
    return good[0] if good else (modes[0] if modes else None)


def _match(mode, modes):
    if not mode or not modes:
        return None
    return min(modes, key=lambda m: abs(m["f"] - mode["f"]))


def _flux_split(sim, regions, f0, Q):
    """Power fraction per named flux group at the mode, averaged over bins within
    f0 +- 3 f0/Q (at least the nearest bin)."""
    mp = _meep()
    out = {}
    for name, fls in regions.items():
        freqs = mp.get_flux_freqs(fls[0])
        tot = None
        for fl in fls:
            v = mp.get_fluxes(fl)
            tot = v if tot is None else [x + y for x, y in zip(tot, v)]
        hw = max(3 * f0 / max(Q, 1.0), 0.0)
        idx = [i for i, f in enumerate(freqs) if abs(f - f0) <= hw]
        if not idx:
            idx = [min(range(len(freqs)), key=lambda i: abs(freqs[i] - f0))]
        out[name] = sum(tot[i] for i in idx) / len(idx)
    return out


def _two_stage_harminv(sim, mp, comp, pts, fcen, df, T, flux_builder=None):
    """comp is one field component for all points, or a list with one per point."""
    """Run to the end of the source, add flux monitors (so they see only the ringdown),
    then run T/2 and T with the same Harminv objects. Returns modes at T/2 and T."""
    sim.init_sim()
    t_src = float(sim.fields.last_source_time())
    t0 = time.time()
    sim.run(until=t_src)
    regions = flux_builder(sim) if flux_builder else None
    comps = comp if isinstance(comp, (list, tuple)) else [comp] * len(pts)
    hs = [mp.Harminv(c, p, fcen, df) for c, p in zip(comps, pts)]
    sim.run(*hs, until=T / 2)
    m_half = _modes(hs, fcen - df / 2, fcen + df / 2)
    sim.run(*hs, until=T / 2)
    m_full = _modes(hs, fcen - df / 2, fcen + df / 2)
    return dict(t_source_end=t_src, T_harm=T, wall_s=time.time() - t0,
                modes_half=m_half, modes=m_full), regions


def _convergence(res_h):
    p = _primary(res_h["modes"])
    ph = _match(p, res_h["modes_half"])
    c = dict(primary=p)
    if p:
        c["Q_over_pi_f_T"] = p["Q"] / (math.pi * p["f"] * res_h["T_harm"])
        if ph:
            c["Q_half"] = ph["Q"]; c["f_half"] = ph["f"]
            c["dQ_rel_half_vs_full"] = abs(ph["Q"] - p["Q"]) / max(p["Q"], 1e-30)
            c["df_rel_half_vs_full"] = abs(ph["f"] - p["f"]) / p["f"]
    return c


# ==========================================================================
# Cylindrical bend-loss Q
# ==========================================================================
def bend_q(p):
    mp = _meep()
    plat = G.platform(p["platform"], t=p.get("t"))
    R, w, t, clad, res, T = p["R"], p["w"], plat["t"], p["clad"], int(p["res"]), float(p["T_after"])
    lam = p.get("lambda_um", LAM0)
    neff = p.get("neff")
    if p.get("m") is None:
        neff = neff or mpb_neff(plat, w, t, clad, lam=lam, res=int(p.get("neff_res", 40)))
        m = int(round(2 * math.pi * R * neff / lam))
    else:
        m = int(p["m"])
    core, ox, bg, _ = _materials(mp, plat, "oxide" if clad == "oxide_fill" else clad)
    dpml, padr, padz = 1.0, p.get("pad_r", 2.5), p.get("pad_z", 1.5)
    sr = R + w / 2 + padr + dpml
    sz = t + 2 * padz + 2 * dpml
    geo = []
    if clad == "sub":
        geo.append(mp.Block(center=mp.Vector3(sr / 2, 0, -t / 2 - sz / 4), size=mp.Vector3(sr, mp.inf, sz / 2), material=ox))
    geo.append(mp.Block(center=mp.Vector3(R, 0, 0), size=mp.Vector3(w, mp.inf, t), material=core))
    fcen, df = 1 / lam, p.get("df_rel", 0.25) / lam
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Er, center=mp.Vector3(R + 0.05 * w, 0, 0.03 * t))]
    sim = mp.Simulation(cell_size=mp.Vector3(sr, 0, sz), geometry=geo, boundary_layers=[mp.PML(dpml)],
                        resolution=res, sources=src, dimensions=mp.CYLINDRICAL, m=m, default_material=bg)
    h_res, _ = _two_stage_harminv(sim, mp, mp.Er, [mp.Vector3(R + 0.05 * w, 0, 0.03 * t)], fcen, df, T)
    conv = _convergence(h_res)
    pm = conv["primary"]
    out = dict(platform=plat["name"], R=R, w=w, t=t, clad=clad, m=m, neff_est=neff, res=res, lambda_um=lam,
               **h_res, convergence=conv,
               Q_bend=pm["Q"] if pm else None, lam_bend_um=pm["lam_um"] if pm else None,
               note="Harminv Q above ~10 pi f T is extrapolated from a short ringdown; treat as a lower-confidence bound")
    return out


# ==========================================================================
# 3D nanobeam
# ==========================================================================
def _geo_holes(mp, holes, t, hole_mat, z0=0.0):
    return [mp.Cylinder(radius=h["r"], height=t, center=mp.Vector3(h["x"], h["y"], z0), material=hole_mat) for h in holes]


def nanobeam3d(p):
    mp = _meep()
    plat = G.platform(p["platform"], t=p.get("t"))
    w, t, clad, res = p["w"], plat["t"], p["clad"], int(p["res"])
    design, dis = p["design"], p.get("disorder")
    nominal = G.build_design(design)
    fab_nom = G.fab_check(nominal, w, plat)
    out = dict(platform=plat["name"], w=w, t=t, clad=clad, res=res, design=design, disorder=dis,
               nholes=len(nominal), fab_nominal=fab_nom)
    if not fab_nom["ok"] and not p.get("allow_fab_violation", False):
        out.update(status="refused_fab", reason="; ".join(fab_nom["violations"]))
        return out
    holes = G.apply_disorder(nominal, dis)
    if dis:
        out["fab_perturbed"] = G.fab_check(holes, w, plat)
    core, ox, bg, hole_mat = _materials(mp, plat, clad)
    dpml = 1.0
    pad_y, pad_z = p.get("pad_y", 1.0), p.get("pad_z", 0.8)
    xmax = max(abs(h["x"]) for h in holes)
    sx = 2 * (xmax + 1.0) + 2 * dpml
    sy = w + 2 * pad_y + 2 * dpml
    sz = t + 2 * pad_z + 2 * dpml
    geo = []
    if clad == "sub":
        geo.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - sz / 4), size=mp.Vector3(mp.inf, mp.inf, sz / 2), material=ox))
    geo.append(mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t), material=core))
    geo += _geo_holes(mp, holes, t, hole_mat)
    fcen, df = float(p["fcen"]), float(p["df"])
    xs_src = p.get("src_x", 0.07)
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey, center=mp.Vector3(xs_src, 0, 0))]
    sym = []
    if not G.breaks_y_mirror(dis):
        sym.append(mp.Mirror(mp.Y, phase=-1))
    if _z_symmetric(clad):
        sym.append(mp.Mirror(mp.Z, phase=+1))
    sim = mp.Simulation(cell_size=mp.Vector3(sx, sy, sz), geometry=geo, sources=src, resolution=res,
                        symmetries=sym, boundary_layers=[mp.PML(dpml)], default_material=bg)
    nfreq = int(p.get("nfreq", 101))
    xb = xmax + 0.5
    yb, zb = sy / 2 - dpml - 0.1, sz / 2 - dpml - 0.1
    win_y, win_z = min(w + 2 * p.get("win_pad", 0.5), 2 * yb), min(t + 2 * p.get("win_pad", 0.5), 2 * zb)

    def flux_builder(sim):
        F = lambda c, s, wgt=1.0: sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=c, size=s, weight=wgt))
        box = [F(mp.Vector3(+xb), mp.Vector3(0, 2 * yb, 2 * zb)), F(mp.Vector3(-xb), mp.Vector3(0, 2 * yb, 2 * zb), -1),
               F(mp.Vector3(0, +yb), mp.Vector3(2 * xb, 0, 2 * zb)), F(mp.Vector3(0, -yb), mp.Vector3(2 * xb, 0, 2 * zb), -1),
               F(mp.Vector3(0, 0, +zb), mp.Vector3(2 * xb, 2 * yb, 0)), F(mp.Vector3(0, 0, -zb), mp.Vector3(2 * xb, 2 * yb, 0), -1)]
        win = [F(mp.Vector3(+xb), mp.Vector3(0, win_y, win_z)), F(mp.Vector3(-xb), mp.Vector3(0, win_y, win_z), -1)]
        return {"total": box, "axial_window": win}

    pts = [mp.Vector3(xs_src, 0, 0), mp.Vector3(-0.21, 0, 0)]
    t0 = time.time()
    h_res, regions = _two_stage_harminv(sim, mp, mp.Ey, pts, fcen, df, float(p["T_harm"]),
                                        flux_builder if p.get("flux", True) else None)
    conv = _convergence(h_res)
    pm = conv["primary"]
    nvox = sx * sy * sz * res ** 3
    sym_factor = 2 ** len(sym)
    nsteps = (h_res["t_source_end"] + h_res["T_harm"]) * 2 * res  # Courant 0.5
    out.update(status="ok", cell=[sx, sy, sz], nvox_full=nvox, symmetry_factor=sym_factor,
               nvox_reduced=nvox / sym_factor, nsteps=nsteps, nproc=mp.count_processors(),
               ns_per_reduced_voxel_step=1e9 * (time.time() - t0) / nsteps / (nvox / sym_factor),
               ns_core_per_reduced_voxel_step=1e9 * (time.time() - t0) * mp.count_processors() / nsteps / (nvox / sym_factor),
               fcen=fcen, df=df, **h_res, convergence=conv)
    if pm and regions:
        P = _flux_split(sim, regions, pm["f"], pm["Q"])
        Pt, Pw = P["total"], P["axial_window"]
        Pr = Pt - Pw
        out["flux"] = dict(P_total=Pt, P_axial_window=Pw, P_rad=Pr,
                           axial_window_size=[win_y, win_z], box_half=[xb, yb, zb])
        if Pt > 0:
            out["Q_total"] = pm["Q"]
            out["Q_axial"] = pm["Q"] * Pt / Pw if Pw > 0 else None
            out["Q_rad"] = pm["Q"] * Pt / Pr if Pr > 0 else None
            out["frac_rad"] = Pr / Pt
    out["peak_rss_GB_rank0"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return out


# ==========================================================================
# Unrolled-ring supercell and full ring
# ==========================================================================
def supercell3d(p):
    mp = _meep()
    plat = G.platform(p["platform"], t=p.get("t"))
    w, t, clad, res = p["w"], plat["t"], p["clad"], int(p["res"])
    a, b, r, N1, N2 = p["a"], p["b"], p["r"], int(p["N1"]), int(p["N2"])
    holes, C, walls = G.periodic_supercell(a, b, r, N1, N2)
    fab = G.fab_check(holes, w, plat, periodic_length=C)
    out = dict(platform=plat["name"], w=w, t=t, clad=clad, res=res, a=a, b=b, r=r, N1=N1, N2=N2,
               circumference_um=C, R_equiv_um=C / (2 * math.pi), walls_x=walls, nholes=len(holes), fab_nominal=fab)
    if not fab["ok"] and not p.get("allow_fab_violation", False):
        out.update(status="refused_fab", reason="; ".join(fab["violations"]))
        return out
    holes = G.apply_disorder(holes, p.get("disorder"))
    core, ox, bg, hole_mat = _materials(mp, plat, clad)
    dpml = 1.0
    pad_y, pad_z = p.get("pad_y", 1.0), p.get("pad_z", 0.8)
    sy, sz = w + 2 * pad_y + 2 * dpml, t + 2 * pad_z + 2 * dpml
    geo = []
    if clad == "sub":
        geo.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - sz / 4), size=mp.Vector3(mp.inf, mp.inf, sz / 2), material=ox))
    geo.append(mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t), material=core))
    geo += _geo_holes(mp, holes, t, hole_mat)
    fcen, df = float(p["fcen"]), float(p["df"])
    xA, xB = walls["A"], walls["B"]
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey, center=mp.Vector3(x + 0.07, 0, 0)) for x in (xA, xB)]
    sym = [mp.Mirror(mp.Y, phase=-1)] + ([mp.Mirror(mp.Z, phase=+1)] if _z_symmetric(clad) else [])
    sim = mp.Simulation(cell_size=mp.Vector3(C, sy, sz), geometry=geo, sources=src, resolution=res, symmetries=sym,
                        boundary_layers=[mp.PML(dpml, direction=mp.Y), mp.PML(dpml, direction=mp.Z)],
                        default_material=bg, k_point=mp.Vector3(0, 0, 0))
    hA = mp.Harminv(mp.Ey, mp.Vector3(xA + 0.07, 0, 0), fcen, df)
    hB = mp.Harminv(mp.Ey, mp.Vector3(xB + 0.07, 0, 0), fcen, df)
    sim.init_sim()
    t_src = float(sim.fields.last_source_time())
    t0 = time.time()
    sim.run(until=t_src)
    T = float(p["T_harm"])
    sim.run(hA, hB, until=T / 2)
    half = {"A": _modes([hA], fcen - df / 2, fcen + df / 2), "B": _modes([hB], fcen - df / 2, fcen + df / 2)}
    sim.run(hA, hB, until=T / 2)
    full = {"A": _modes([hA], fcen - df / 2, fcen + df / 2), "B": _modes([hB], fcen - df / 2, fcen + df / 2)}
    out.update(status="ok", t_source_end=t_src, T_harm=T, wall_s=time.time() - t0, modes_half=half, modes=full,
               primary_A=_primary(full["A"]), primary_B=_primary(full["B"]), nproc=mp.count_processors(),
               nvox_full=C * sy * sz * res ** 3, symmetry_factor=2 ** len(sym),
               note="periodic, straight: no bend loss; Q is radiation only. A/B wall detuning = primary_A vs primary_B")
    return out


def ring3d(p):
    mp = _meep()
    plat = G.platform(p["platform"], t=p.get("t"))
    w, t, clad, res = p["w"], plat["t"], p["clad"], int(p["res"])
    a, b, r, N1, N2 = p["a"], p["b"], p["r"], int(p["N1"]), int(p["N2"])
    gaps = G.ring_gap_sequence(a, b, N1, N2)
    C = sum(gaps); R = C / (2 * math.pi)
    s = [0.0]
    for g in gaps[:-1]:
        s.append(s[-1] + g)
    # fab check on the unrolled arc-length sequence (curvature changes webs by O(r^2/R))
    flat = [dict(x=v, y=0.0, r=r, ord=i) for i, v in enumerate(s)]
    fab = G.fab_check(flat, w, plat, periodic_length=C)
    out = dict(platform=plat["name"], w=w, t=t, clad=clad, res=res, a=a, b=b, r=r, N1=N1, N2=N2, R_um=R,
               nholes=len(s), fab_nominal=fab)
    if not fab["ok"] and not p.get("allow_fab_violation", False):
        out.update(status="refused_fab", reason="; ".join(fab["violations"]))
        return out
    th = [v / R for v in s]
    thA, thB = th[0], th[2 * N1]
    core, ox, bg, hole_mat = _materials(mp, plat, clad)
    dpml = 1.0
    sxy = 2 * (R + w / 2 + p.get("pad_r", 1.0)) + 2 * dpml
    sz = t + 2 * p.get("pad_z", 0.8) + 2 * dpml
    inner = ox if clad in ("oxide", "oxide_fill") else mp.air
    geo = []
    if clad == "sub":
        geo.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - sz / 4), size=mp.Vector3(mp.inf, mp.inf, sz / 2), material=ox))
    geo += [mp.Cylinder(radius=R + w / 2, height=t, material=core), mp.Cylinder(radius=R - w / 2, height=t, material=inner)]
    geo += [mp.Cylinder(radius=r, height=t, center=mp.Vector3(R * math.cos(q), R * math.sin(q), 0), material=hole_mat) for q in th]
    fcen, df = float(p["fcen"]), float(p["df"])

    def pt(q, dr=0.0):
        return mp.Vector3((R + dr) * math.cos(q + 0.07 / R), (R + dr) * math.sin(q + 0.07 / R), 0)
    # TE-like ring mode: dominant E is radial; at angle q pick Ex if |cos q| > |sin q| else Ey
    def comp(q):
        return mp.Ex if abs(math.cos(q)) >= abs(math.sin(q)) else mp.Ey
    src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=comp(thA), center=pt(thA))]
    sym = [mp.Mirror(mp.Z, phase=+1)] if _z_symmetric(clad) else []
    sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy, sz), geometry=geo, sources=src, resolution=res,
                        boundary_layers=[mp.PML(dpml)], default_material=bg, symmetries=sym)
    h_res, _ = _two_stage_harminv(sim, mp, [comp(thA), comp(thB)], [pt(thA), pt(thB)], fcen, df, float(p["T_harm"]))
    out.update(status="ok", walls_theta={"A": thA, "B": thB}, **h_res, convergence=_convergence(h_res),
               nproc=mp.count_processors(), nvox_full=sxy * sxy * sz * res ** 3,
               peak_rss_GB_rank0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
               note="source at the A wall, Harminv at both walls; total Q includes bend, wall radiation and wall-wall coupling")
    return out


KERNELS = dict(mpb_cell=mpb_cell, bend_q=bend_q, nanobeam3d=nanobeam3d, supercell3d=supercell3d, ring3d=ring3d)


def _json_default(o):
    if isinstance(o, complex):
        return abs(o)
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def main(argv):
    kernel, pin, pout = argv[1:4]
    p = json.load(open(pin))
    t0 = time.time()
    res = KERNELS[kernel](p)
    import meep as mp
    if mp.am_master():
        res["kernel"] = kernel
        res["kernel_wall_s"] = time.time() - t0
        res["meep_version"] = getattr(mp, "__version__", "?")
        with open(pout, "w") as f:
            json.dump(res, f, indent=1, default=_json_default)


if __name__ == "__main__":
    main(sys.argv)
