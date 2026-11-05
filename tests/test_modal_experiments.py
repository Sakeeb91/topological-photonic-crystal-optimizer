"""Unit tests for research/modal (geometry, fab rules, CRN disorder, manifests, keys).
No MEEP and no Modal needed."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "research", "modal"))
import tpc_geometry as G  # noqa: E402
import tpc_plan as P  # noqa: E402


def xs(holes):
    return [round(h["x"], 9) for h in holes]


def test_ssh_wall_matches_pilot_convention():
    # pilot c_meep3d_nanobeam.py, wall A: centre hole, then gaps a, b, a, b outward
    h = G.ssh_nanobeam(0.35, 0.15, 0.1, Nside=1, wall="A")
    assert xs(h) == [-0.5, -0.35, 0.0, 0.35, 0.5]
    hB = G.ssh_nanobeam(0.35, 0.15, 0.1, Nside=1, wall="B")
    assert xs(hB) == [-0.5, -0.15, 0.0, 0.15, 0.5]
    hn = G.ssh_nanobeam(0.35, 0.15, 0.1, Nside=2, wall="none")
    assert len(hn) == 8 and np.allclose(np.diff(xs(hn)), [0.35, 0.15] * 3 + [0.35])


def test_tanh_wall_tends_to_bulk_dimerization():
    a, b = 0.30, 0.20
    h = G.ssh_nanobeam(a, b, 0.05, Nside=30, wall="tanh", M=3)
    gaps = np.diff([v for v in xs(h) if v >= 0])
    assert abs(gaps[-1] - a) < 1e-4 and abs(gaps[-2] - b) < 1e-4
    assert abs(gaps[0] - gaps[1]) < (a - b)  # weak dimerization at the wall


def test_quan_loncar_profile():
    h = G.quan_loncar_nanobeam(0.4, 0.08, 0.12, n_taper=5, n_mirror=3)
    pos = [q for q in h if q["x"] > 0]
    assert len(pos) == 8 and abs(pos[0]["x"] - 0.2) < 1e-12
    rs = [q["r"] for q in pos]
    assert rs[0] == pytest.approx(0.08) and rs[-1] == pytest.approx(0.12)
    assert all(np.diff(rs) >= -1e-12)


def test_fab_rules_web_is_gap_minus_diameter():
    plat = G.platform("si220")
    ok = G.ssh_cell_fab_check(0.20, 0.16, 0.035, 0.5, plat)  # webs 130 and 90 nm, d = 70 nm
    assert ok["ok"] and ok["min_web_nm"] == pytest.approx(90.0)
    bad = G.ssh_cell_fab_check(0.35, 0.15, 0.10, 0.5, plat)  # pilot geometry: web -50 nm
    assert not bad["ok"] and bad["min_web_nm"] == pytest.approx(-50.0)
    assert any("web" in v for v in bad["violations"])
    sin = G.platform("sin400")
    assert not G.ssh_cell_fab_check(0.20, 0.16, 0.035, 0.5, sin)["ok"]  # 120 nm rule


def test_pilot_3d_geometries_are_refused():
    # every pilot 3D domain-wall geometry had webs of 10 nm down to -50 nm
    for a, b, r in [(0.35, 0.15, 0.10), (0.35, 0.15, 0.07), (0.30, 0.20, 0.10), (0.27, 0.23, 0.10)]:
        h = G.ssh_nanobeam(a, b, r, Nside=5, wall="A")
        assert not G.fab_check(h, 0.5, G.platform("si220"))["ok"]


def test_disorder_common_random_numbers_and_class_independence():
    hA = G.ssh_nanobeam(0.20, 0.16, 0.035, Nside=10, wall="A")
    hT = G.ssh_nanobeam(0.20, 0.16, 0.035, Nside=10, wall="tanh", M=4)
    spec = dict(seed=7, realization=3, sigma_r_nm=5.0)
    dA = {h["ord"]: h["r"] - 0.035 for h in G.apply_disorder(hA, spec)}
    dT = {h["ord"]: h["r"] - 0.035 for h in G.apply_disorder(hT, spec)}
    assert all(dA[k] == pytest.approx(dT[k]) for k in dA)  # same draw per hole across designs
    spec2 = dict(spec, sigma_x_nm=5.0)
    d2 = {h["ord"]: h["r"] - 0.035 for h in G.apply_disorder(hA, spec2)}
    assert all(dA[k] == pytest.approx(d2[k]) for k in dA)  # adding a class keeps radius draws
    allr = np.array(list(dA.values())) * 1e3
    assert 1.0 < allr.std() < 10.0  # absolute nm, sigma 5
    bias = G.apply_disorder(hA, dict(seed=7, realization=3, sigma_bias_nm=4.0))
    db = {round(h["r"] - 0.035, 12) for h in bias}
    assert len(db) == 1  # global CD bias: one draw for all holes
    assert G.breaks_y_mirror(dict(sigma_y_nm=1.0)) and not G.breaks_y_mirror(spec)


def test_supercell_has_one_A_and_one_B_wall():
    holes, C, walls = G.periodic_supercell(0.20, 0.16, 0.035, 10, 10)
    assert C == pytest.approx(20 * 0.36)
    x = sorted(h["x"] for h in holes)
    gaps = np.diff(x + [x[0] + C])
    pairs = [(round(gaps[i - 1], 6), round(gaps[i], 6)) for i in range(len(gaps))]
    assert pairs.count((0.2, 0.2)) == 1 and pairs.count((0.16, 0.16)) == 1
    assert G.fab_check(holes, 0.5, G.platform("si220"), periodic_length=C)["ok"]


def test_job_keys_are_stable_and_param_sensitive():
    j1 = P.make_job("E1", "bend_q", dict(platform="si220", R=5.0, w=0.5, clad="oxide", res=40, T_after=2000))
    j2 = P.make_job("E1", "bend_q", dict(T_after=2000, res=40, clad="oxide", w=0.5, R=5.0, platform="si220"))
    j3 = P.make_job("E1", "bend_q", dict(platform="si220", R=5.0, w=0.5, clad="oxide", res=30, T_after=2000))
    assert j1["key"] == j2["key"] != j3["key"]
    assert P.job_path(j1) == f"E1/bend_q/{j1['key']}.json"


@pytest.mark.parametrize("exp,n", [("E0", 2), ("E1", 40), ("E2", 160)])
def test_manifests_expand(exp, n):
    jobs = P.expand(P.load_manifest(exp))
    assert len(jobs) == n and len({j["key"] for j in jobs}) == n
    s = P.summarize_cost(jobs)
    assert s["usd"] > 0 and np.isfinite(s["container_hours"])


def _fake_e2():
    plat = G.platform("si220")
    a, b, r = 0.20, 0.16, 0.035
    return [dict(kernel="mpb_cell", platform="si220", clad="sub", w=0.5, a=a, b=b, r=r, L=a + b,
                 edge_f1=0.640, edge_f2=0.652, gap_abs=0.012, gap_mid_ratio=0.0186,
                 gap_contains_target=True, below_lightline=True, fab=G.ssh_cell_fab_check(a, b, r, 0.5, plat))]


def test_selection_manifests_expand_from_e2():
    e3 = P.expand(P.load_manifest("E3"), lambda e: _fake_e2())
    assert len(e3) == 8 and {j["tier"] for j in e3} <= {"3d", "3d_big"}
    e4 = P.expand(P.load_manifest("E4"), lambda e: _fake_e2())
    assert len(e4) == 3 * (1 + 4 * 2 * 4)  # 3 designs x (nominal + 4 classes x 2 sigmas x N=4)
    e5 = P.expand(P.load_manifest("E5"), lambda e: _fake_e2())
    assert [j["kernel"] for j in e5] == ["supercell3d", "supercell3d", "ring3d"]
    ring = e5[-1]
    assert ring["tier"] == "ring" and P.cost(ring)["hours"] < 24


def test_ensemble_stats():
    rows = [dict(Q_total=q, Q_rad=2 * q, convergence=dict(primary=dict(lam_um=1.547 + 1e-3 * i)))
            for i, q in enumerate([1000, 2000, 4000, 500])]
    s = P.ensemble_stats(rows)
    assert s["n"] == 4 and s["Q_harmonic"] == pytest.approx(4 / (1 / 1000 + 1 / 2000 + 1 / 4000 + 1 / 500))
    assert s["Q10"] <= s["Q_median"]
