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


@pytest.mark.parametrize("exp,n", [("E0", 2), ("E1", 44), ("E2", 160)])
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
    # 3 SSH walls x 3 Nside + Quan-Loncar 2 n_mirror x 3 period_scale, one resolution, + parity check
    assert len(e3) == 16 and {j["tier"] for j in e3} <= {"light", "3d", "3d_big", "3d_xl"}
    assert sum(j["params"].get("check_other_parity", False) for j in e3) == 1
    nb = [j for j in e3 if j["kernel"] == "nanobeam3d"]
    assert all(j["params"]["res"] * min(h["r"] for h in G.build_design(j["params"]["design"])) >= 1.5 for j in nb)
    e3c = P.expand(P.load_manifest(os.path.join(os.path.dirname(P.__file__), "manifests", "e3c.yaml")), lambda e: _fake_e2())
    assert len({j["params"]["res"] for j in e3c if j["tag"].startswith("ssh_A")}) == 3
    e4 = P.expand(P.load_manifest("E4"), lambda e: _fake_e2())
    assert len(e4) == 1 + 3 * (1 + 4 * 2 * 4)  # + 1 parity check  # 3 designs x (nominal + 4 classes x 2 sigmas x N=4)
    e5 = P.expand(P.load_manifest("E5"), lambda e: _fake_e2())
    assert [j["kernel"] for j in e5] == ["mpb_cell", "supercell3d", "supercell3d", "ring3d"]
    ring = e5[-1]
    assert ring["tier"] == "ring" and P.cost(ring)["hours"] < 24


def test_ensemble_stats():
    rows = [dict(Q_total=q, Q_rad=2 * q, convergence=dict(primary=dict(lam_um=1.547 + 1e-3 * i)))
            for i, q in enumerate([1000, 2000, 4000, 500])]
    s = P.ensemble_stats(rows)
    assert s["n"] == 4 and s["Q_harmonic"] == pytest.approx(4 / (1 / 1000 + 1 / 2000 + 1 / 4000 + 1 / 500))
    assert s["Q10"] <= s["Q_median"]


def _bq(platform, w, R, res, Q, dq=0.01, lam=1.55):
    return dict(kernel="bend_q", platform=platform, clad="oxide", w=w, R=R, res=res, T_harm=2000, Q_bend=Q,
                lam_bend_um=lam, convergence=dict(dQ_rel_half_vs_full=dq))


def test_e1_gonogo_saturates_at_q_cap():
    # the four convergence pairs reported from the first E1 run on Modal (2026-11-05)
    rows = [_bq("sin400", 1.0, 10.0, 30, 2022), _bq("sin400", 1.0, 10.0, 45, 1743),
            _bq("sin400", 1.0, 20.0, 30, 6.48e5), _bq("sin400", 1.0, 20.0, 45, 4.72e5),
            _bq("si220", 0.5, 3.0, 40, 1.22e9), _bq("si220", 0.5, 3.0, 60, 7.99e8),
            _bq("si220", 0.5, 5.0, 40, 4.4e10, dq=45), _bq("si220", 0.5, 5.0, 60, 1.9e8, dq=7.3e3)]
    g = P.gonogo("E1", rows)
    v = {k: e["verdict"] for k, e in g["resolution_convergence"].items()}
    assert v["sin400|oxide|w1.0|R10.0"] == "agree"
    assert v["sin400|oxide|w1.0|R20.0"] == "same_side_of_threshold"
    assert v["si220|oxide|w0.5|R3.0"] == "agree" and v["si220|oxide|w0.5|R5.0"] == "agree"
    assert g["go"] and g["smallest_R_with_Qbend_ge_1e5"]["si220"]["R"] == 3.0
    # a negative-Q row with no above-cap partner is flagged
    bad = P.gonogo("E1", [_bq("sin400", 0.8, 10.0, 30, -5.0), _bq("sin400", 0.8, 10.0, 45, 900)])
    assert not bad["go"] and bad["flagged_pairs"]
    # negative Q paired with a positive above-cap value: agree as > cap
    neg = P.gonogo("E1", [_bq("si220", 0.45, 2.0, 40, -3.0), _bq("si220", 0.45, 2.0, 60, 5e7, dq=2.0)])
    assert neg["resolution_convergence"]["si220|oxide|w0.45|R2.0"]["verdict"] == "agree"


def test_select_cells_margin():
    rows = _fake_e2()  # min web 90 nm vs the 60 nm Si rule -> margin 30 nm (diameter 70 vs 60 -> 10 nm)
    assert len(P.select_cells({"platform": "si220", "min_margin_nm": 10}, rows)) == 1
    assert len(P.select_cells({"platform": "si220", "min_margin_nm": 11}, rows)) == 0


def _nb(design, res, Q_rad, Q_axial, lam, fcen=0.646, df=0.04, name="ssh_A"):
    return dict(kernel="nanobeam3d", status="ok", design=design, res=res, Q_rad=Q_rad, Q_axial=Q_axial,
                Q_total=1 / (1 / Q_rad + 1 / Q_axial), fcen=fcen, df=df, T_harm=1000,
                convergence=dict(primary=dict(lam_um=lam)), job=dict(tag=f"{name}|res{res}|nominal"))


def test_e3_gonogo_groups_by_cell_and_flags_mirror_limit():
    d1 = dict(kind="ssh", a=0.2225, b=0.1335, r=0.0356, Nside=30, wall="A")
    d2 = dict(d1, a=0.21, b=0.15)  # second cell: must not overwrite the first
    rows = [_nb(d1, r, 1e5 * (1 + 0.01 * i), 2e6, 1.547 + 1e-4 * i) for i, r in enumerate((43, 57, 71))]
    rows += [_nb(d2, r, 1e5, 5e4, 1.55) for r in (43, 57, 71)]  # mirror-limited
    g = P.gonogo("E3", rows)
    assert len(g["variants"]) == 2
    u = g["design_usable"]
    assert u["L356.0nm|ssh_A"] is True and u["L360.0nm|ssh_A"] is False
    assert not g["go"]  # no Quan-Loncar control yet


SNAPSHOT_E3 = os.path.join(os.path.dirname(__file__), "..", "research", "results", "modal_volume_2026-11-05", "E3")


@pytest.mark.skipif(not os.path.isdir(SNAPSHOT_E3), reason="Modal results snapshot not present")
def test_e3_report_on_modal_snapshot_keeps_both_cells():
    # issue 1 in docs/EXPERIMENT_LOG.md: the first E3 run (top_k 2) had two cells whose results
    # collapsed to 4 entries when keyed by design name only
    import glob
    import json
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(SNAPSHOT_E3, "**", "*.json"), recursive=True)]
    # the snapshot grows as later runs land; this test covers the first-run design kinds
    rows = [r for r in rows if (r.get("design") or {}).get("kind", "ssh") in ("ssh", "quan_loncar")]
    g = P.gonogo("E3", rows)
    cells = {v["cell"] for v in g["variants"].values()}
    assert len(cells) >= 2
    first = {k: v for k, v in g["variants"].items()
             if ("Nside=20" in k or "quan_loncar" in k) and {20, 30} <= set(v["by_res"])}
    # the first run: 2 cells x (A, B, tanh at Nside 20, Quan-Loncar), each at res 20 and 30
    assert sum(1 for k in first if set(first[k]["by_res"]) >= {20, 30}) == 8
    assert g["go"] is False  # not converged over 3 resolutions; mirror-limited
    assert all(first[k]["mirror_limited"] for k in first)
