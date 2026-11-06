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


def _fake_e2_bighole():
    # an e2b-like cell: r = 53.5 nm, webs 95 / 73 nm (>= 10 nm margin on the 60 nm rule)
    plat = G.platform("si220")
    a, b, r = 0.2025, 0.1796, 0.0535
    return [dict(kernel="mpb_cell", platform="si220", clad="sub", w=0.5, a=a, b=b, r=r, L=a + b,
                 edge_f1=0.640, edge_f2=0.652, gap_abs=0.012, gap_mid_ratio=0.0186,
                 gap_contains_target=True, below_lightline=True, fab=G.ssh_cell_fab_check(a, b, r, 0.5, plat))]


def _recipe_hits(rec_jobs):
    """Fake ql_recipe results shaped like the local production-res runs (2026-11-06, res 32):
    air 461.6 nm period, radii 96.9 -> 141.9 nm; dielectric 357.4 nm, 107.2 -> 80.9 nm."""
    out = []
    for j in rec_jobs:
        p = j["params"]
        L, r0, r1 = (0.4616, 0.0969, 0.1419) if p["mode_type"] == "air" else (0.3574, 0.1072, 0.0809)
        radii = list(np.linspace(r0, r1, p["n_taper"]))
        out.append(dict(kernel="ql_recipe", converged=True, fab_ok=True, period=L, radii=radii, job=dict(key=j["key"])))
    return out


def test_selection_manifests_expand_from_e2():
    assert P.expand(P.load_manifest("E3"), lambda e: _fake_e2()) == []  # r = 35 nm fails min_r_nm 50
    e3 = P.expand(P.load_manifest("E3"), lambda e: _fake_e2_bighole())
    # 3 SSH walls x 3 Nside + graded x 2 Nside at one resolution + parity check + mirror_cell_check
    # + 4 ql_recipe MPB jobs (air/dielectric x n_taper 20/30; QL nanobeams wait for them)
    assert len(e3) == 17 and {j["tier"] for j in e3} <= {"light", "3d", "3d_big", "3d_xl"}
    assert sum(j["params"].get("check_other_parity", False) for j in e3) == 1
    rec = [j for j in e3 if j["kernel"] == "ql_recipe"]
    assert sorted((j["params"]["mode_type"], j["params"]["n_taper"]) for j in rec) == \
        [("air", 20), ("air", 30), ("dielectric", 20), ("dielectric", 30)]
    e3b = P.expand(P.load_manifest("E3"), lambda e: _fake_e2_bighole() if e == "E2" else _recipe_hits(rec))
    ql = [j for j in e3b if j["kernel"] == "nanobeam3d" and j["tag"].startswith("quan_loncar")]
    assert len(ql) == 4 and all(len(j["params"]["design"]["radii"]) == j["params"]["design"]["n_taper"] for j in ql)
    # the older tune: true path (ql_period) still works for manifests that use it
    tm = dict(P.load_manifest("E3"), designs=[dict(kind="quan_loncar", name="ql_old", tune=True, n_taper=10, n_mirror=15)])
    assert [j["kernel"] for j in P.expand(tm, lambda e: _fake_e2_bighole() if e == "E2" else []) if j["kernel"] != "mpb_cell"] == ["ql_period"]
    nb = [j for j in e3 if j["kernel"] == "nanobeam3d"]
    assert all(j["params"]["res"] * min(h["r"] for h in G.build_design(j["params"]["design"])) >= 1.5 for j in nb)
    e3c = P.expand(P.load_manifest(os.path.join(os.path.dirname(P.__file__), "manifests", "e3c.yaml")), lambda e: _fake_e2())
    assert len({j["params"]["res"] for j in e3c if j["tag"].startswith("ssh_A")}) == 3
    assert P.expand(P.load_manifest("E4"), lambda e: _fake_e2()) == []  # r = 35 nm fails min_r_nm 50
    e4 = P.expand(P.load_manifest("E4"), lambda e: _fake_e2_bighole())
    # parity check + 3 designs x (nominal + sens3d + 4 classes x 1 sigma x N=1)  (budget default)
    assert len(e4) == 1 + 3 * (2 + 4 * 1 * 1)
    assert sum(j["kernel"] == "sens3d" for j in e4) == 3  # 3 designs x (nominal + 4 classes x 2 sigmas x N=4)
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


FIX_PT = os.path.join(os.path.dirname(__file__), "fixtures", "pt_validation_res20.json")


def test_pt_shift_and_stats_on_validation_runs():
    """sens3d gradients vs FDTD on the same CRN realizations (validation scale, res 20)."""
    import json
    fx = json.load(open(FIX_PT))
    sens, nom = fx["sens"], fx["nominal"]
    f_nom = nom["convergence"]["primary"]["f"]
    pairs = []
    for k, r in fx["spots"].items():
        fd = min([r["convergence"]["primary"]], key=lambda m: abs(m["f"] - sens["f0"]))["f"] - f_nom
        pairs.append((fd, P.pt_shift(sens, sens["design"], r["disorder"])))
    a = np.array(pairs)
    assert np.all(np.sign(a[:, 0]) == np.sign(a[:, 1]))  # same sign in every realization
    slope = (a[:, 0] @ a[:, 1]) / (a[:, 1] @ a[:, 1])
    assert 0.5 < slope < 2.0
    st = P.pt_stats(sens, sens["design"], ["radius", "position_x", "cd_bias"], [5], n_draws=400)
    for v in st["by_class"].values():  # linear model: Monte Carlo matches the analytic value
        assert v["sigma_lambda_nm_mc"] == pytest.approx(v["sigma_lambda_nm_analytic"], rel=0.15)


def test_group_ensemble_reports_pt_and_minkov_c():
    import json
    fx = json.load(open(FIX_PT))
    d = fx["sens"]["design"]
    base = dict(platform="sin400", w=0.6, clad="susp", res=20, fcen=0.8, df=0.2, T_harm=150, design=d)
    jobs, rows = {}, []
    def add(tag, kernel, r, dis=None):
        if kernel == "nanobeam3d":  # these validation runs had flux off; use the Harminv Q as Q_total
            r = dict(r, Q_total=r["convergence"]["primary"]["Q"])
        j = P.make_job("E4", kernel, dict(base, **({"disorder": dis} if dis else {})), tag)
        jobs[j["key"]] = j; rows.append((j["key"], r))
    add("ssh_A|res20|nominal", "nanobeam3d", fx["nominal"])
    add("ssh_A|res20|sens", "sens3d", fx["sens"])
    for i, k in enumerate(["x10a", "x10b"]):
        add(f"ssh_A|res20|position_x|s10|k{i}", "nanobeam3d", fx["spots"][k], fx["spots"][k]["disorder"])
    out = P.group_ensemble(rows, jobs, pt=dict(classes=["radius", "position_x"], sigmas_nm=[1, 3], n_draws=200))
    cell = out["ssh_A|res20|position_x|s10"]
    assert cell["pt_check"]["n"] == 2 and "C_minkov_per_nm2" in cell
    assert set(out["ssh_A|res20|pt"]["by_class"]) == {"radius|s1", "radius|s3", "position_x|s1", "position_x|s3"}


def test_budget_guard():
    items = [dict(description="tpc-experiments", cost="3.10"), dict(description="ssh-ring-smoke", cost="0.82"),
             dict(description="hf-physics-x", cost="40.0")]  # the user's other apps do not count
    spent = P.program_spend(items)
    assert spent == pytest.approx(3.92)
    assert P.budget_check(spent, 3.0)["ok"]                     # 3.92 + 1.6 x 3.0 = 8.72
    assert not P.budget_check(spent, 4.0)["ok"]                 # 3.92 + 1.6 x 4.0 = 10.32
    assert P.budget_check(spent, 4.0, cap_usd=12.0)["ok"]


def test_e4_reuses_e3_mode_for_sens3d():
    cell = _fake_e2_bighole()
    m = P.load_manifest("E4")
    e4 = P.expand(m, lambda e: cell)
    s = [j for j in e4 if j["kernel"] == "sens3d"][0]
    assert "f_mode" not in s["params"]  # no E3 results yet: two-pass sens3d
    e3row = dict(kernel="nanobeam3d", status="ok", platform=s["params"]["platform"], w=s["params"]["w"],
                 clad=s["params"]["clad"], res=s["params"]["res"], design=s["params"]["design"],
                 convergence=dict(primary=dict(f=0.6461, Q=4321.0)), job=dict(key="abc"))
    e4b = P.expand(m, lambda e: cell if e == "E2" else [e3row])
    s2 = [j for j in e4b if j["kernel"] == "sens3d"][0]
    assert s2["params"]["f_mode"] == 0.6461 and s2["params"]["f_mode_from"] == "abc"
    assert P.cost(s2)["hours"] < P.cost(s)["hours"]  # one pass instead of two


def test_graded_mirror_design():
    plat = G.platform("si220")
    L, r = 0.3821, 0.0535
    dm = G.max_delta(L, r, plat, margin_nm=5)
    assert 0.09 < dm < 0.11
    h = G.graded_nanobeam(L, r, Nside=30, delta_wall=0.0, delta_mirror=dm, M=4)
    fab = G.fab_check(h, 0.5, plat)
    assert fab["ok"] and fab["min_web_nm"] == pytest.approx(65.0, abs=0.5)  # 60 nm rule + 5 nm margin
    x = sorted(q["x"] for q in h)
    gaps = np.diff(x)
    centre = len(gaps) // 2
    assert abs(gaps[centre] - gaps[centre + 1]) < 0.02 * L          # weak dimerization at the wall
    assert abs(abs(gaps[-1] - gaps[-2]) - dm * L) < 1e-3 * L          # full mirror dimerization far out
    e3 = P.expand(P.load_manifest("E3"), lambda e: _fake_e2_bighole() if e == "E2" else [])
    gr = [j for j in e3 if j["tag"].startswith("ssh_graded")]
    assert len(gr) == 2 and all(j["params"]["design"]["delta_mirror"] > 0.06 for j in gr)
    assert sum(j["tag"] == "mirror_cell_check" for j in e3) == 1


def test_e3_q_rad_plateau_flag():
    d = dict(kind="ssh", a=0.2022, b=0.1793, r=0.0534, wall="A")
    rows = [_nb(dict(d, Nside=n), 29, q, 1e6, 1.547, name=f"ssh_A[Nside={n}]") for n, q in ((30, 1.24e5), (40, 2.81e5))]
    g = P.gonogo("E3", rows)
    p = g["q_rad_plateau"]["L381.5nm|ssh_A"]
    assert p["nside"] == [30, 40] and not p["plateaued"] and p["dQrad_rel"] > 0.5
    rows2 = [_nb(dict(d, Nside=n), 29, q, 1e7, 1.547, name=f"ssh_A[Nside={n}]") for n, q in ((40, 2.6e5), (60, 2.8e5))]
    assert P.gonogo("E3", rows2)["q_rad_plateau"]["L381.5nm|ssh_A"]["plateaued"]


@pytest.mark.skipif(not os.path.isdir(SNAPSHOT_E3), reason="Modal results snapshot not present")
def test_e3_bighole_snapshot_not_plateaued():
    import glob
    import json
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(SNAPSHOT_E3, "**", "*.json"), recursive=True)]
    pl = P.gonogo("E3", rows)["q_rad_plateau"]
    big = {k: v for k, v in pl.items() if k.startswith("L381.5nm|ssh_")}
    if not big:
        pytest.skip("snapshot predates E3-bighole")
    assert all(not v["plateaued"] for v in big.values())  # Nside 30 -> 40 still moves Q_rad by > 20%


def test_local_backend_planning(tmp_path):
    import json
    import run_local as RL
    job = P.make_job("E2", "mpb_cell", dict(platform="si220", clad="sub", w=0.5, res=40, L0=0.36, delta=0.1,
                                            r_over_L=0.1, target_lambda_um=1.547))
    assert job["tier"] == "light" and RL.local_hours(job, 4) < 0.1
    d = tmp_path / "E2" / "mpb_cell"
    d.mkdir(parents=True)
    (d / f"{job['key']}.json").write_text(json.dumps(dict(kernel="mpb_cell", status="ok")))
    assert job["key"] in RL.existing_keys([str(tmp_path)], "E2")
    assert len(RL.make_loader([str(tmp_path)])("E2")) == 1
    big = P.expand(P.load_manifest("E3"), lambda e: _fake_e2_bighole() if e == "E2" else [])
    nb = [j for j in big if j["kernel"] == "nanobeam3d"]
    assert all(RL.local_hours(j, 4) > 0.5 for j in nb)  # E3 3D jobs go to Modal, not the 4-core box


class _FakeBilling:
    """Mimics modal 1.6.3 billing.report as measured on 2026-11-05: daily rows omit the in-progress
    day; hourly rows include the in-progress hour only when `end` is passed past it."""

    def __init__(self, now):
        from decimal import Decimal
        from types import SimpleNamespace as NS
        self.now = now
        h = now.replace(minute=0, second=0, microsecond=0)
        self.rows = [NS(description="tpc-experiments", cost=Decimal("4.682"), interval_start=h.replace(hour=h.hour - 2)),
                     NS(description="tpc-experiments", cost=Decimal("1.951"), interval_start=h),  # in-progress hour
                     NS(description="hf-physics-x", cost=Decimal("9.0"), interval_start=h)]
        self.calls = []

    def report(self, *, start, end=None, resolution="d", tag_names=None):
        self.calls.append(dict(start=start, end=end, resolution=resolution))
        if resolution == "d":
            return []  # in-progress day omitted
        cutoff = end if end is not None else self.now.replace(minute=0, second=0, microsecond=0)
        return [r for r in self.rows if r.interval_start < cutoff]


def test_read_program_spend_includes_current_hour_and_fails_closed():
    from datetime import datetime, timezone
    now = datetime(2026, 11, 5, 22, 31, tzinfo=timezone.utc)
    fb = _FakeBilling(now)
    assert P.read_program_spend(fb, now=now) == pytest.approx(6.633)  # matches the CLI figure
    assert fb.calls[-1]["resolution"] == "h" and fb.calls[-1]["end"] > now
    assert P.program_spend(fb.report(start=now, resolution="h")) == pytest.approx(4.682)  # the old bug
    empty = _FakeBilling(now); empty.rows = []
    with pytest.raises(P.SpendUnknown):
        P.read_program_spend(empty, now=now)


def test_common_resolution_and_floor():
    assert P.res_floor("si220") == 23 and P.res_floor("sin400") == 13
    # tuned QL control from E3 v2 (r_c = 96.9 nm) alone would get ceil(1.5 / 0.0969) = 16 -> NaN on Modal
    rec = [j for j in P.expand(P.load_manifest("E3"), lambda e: _fake_e2_bighole() if e == "E2" else [])
           if j["kernel"] == "ql_recipe"]
    hit = _recipe_hits(rec)
    e3 = P.expand(P.load_manifest("E3"), lambda e: _fake_e2_bighole() if e == "E2" else hit)
    nb = [j for j in e3 if j["kernel"] == "nanobeam3d"]
    assert {j["params"]["res"] for j in nb} == {29}  # SSH holes (r 53.5 nm) set it for every design on the cell
    assert any(j["tag"].startswith("quan_loncar") for j in nb)
    alone = dict(P.load_manifest("E3"), designs=[d for d in P.load_manifest("E3")["designs"] if d["kind"] == "quan_loncar"])
    ql_only = [j for j in P.expand(alone, lambda e: _fake_e2_bighole() if e == "E2" else hit) if j["kernel"] == "nanobeam3d"]
    assert {j["params"]["res"] for j in ql_only} == {23}  # floor, not ceil(1.5 / r_min) = 16-19


def test_unstable_runs_are_recorded_and_reported():
    job = P.make_job("E3", "nanobeam3d", dict(platform="si220", w=0.5, clad="sub", res=16, fcen=0.646, df=0.04,
                                              T_harm=1000, design=dict(kind="quan_loncar", period=0.4616, r_center=0.0969,
                                                                       r_end=0.1385, n_taper=10, n_mirror=15)),
                     tag="quan_loncar[n_mirror=15]|res16|nominal")
    err = "meep: simulation fields are NaN or Inf\nMPI_ABORT was invoked"
    u = P.unstable_result(job, "", err, 104.0)
    assert u["status"] == "unstable" and u["res"] == 16
    assert P.unstable_result(job, "", "Killed (OOM)", 5.0) is None  # other failures stay in _failed
    g = P.gonogo("E3", [u])
    assert g["unstable"] == [dict(tag="quan_loncar[n_mirror=15]|res16|nominal", res=16)] and g["go"] is False


def test_quan_loncar_explicit_radii_and_mirror_strength():
    import tpc_physics as T
    h = G.build_design(dict(kind="quan_loncar", period=0.4616, radii=[0.0969, 0.12, 0.1419], n_taper=3, n_mirror=2))
    pos = sorted([q for q in h if q["x"] > 0], key=lambda q: q["x"])
    assert [round(q["r"], 4) for q in pos] == [0.0969, 0.12, 0.1419, 0.1419, 0.1419]
    assert abs(pos[0]["x"] - 0.2308) < 1e-9
    assert T._mirror_strength(0.60, 0.70, 0.60) == pytest.approx(0.0, abs=1e-6)    # at the band edge
    assert T._mirror_strength(0.60, 0.70, 0.65) == pytest.approx(0.1 / 1.3)        # mid-gap: half-gap / mid
    assert T._mirror_strength(0.60, 0.70, 0.75) == 0.0                             # outside the gap


def test_e3_limiting_channel_and_comparability():
    d = dict(kind="ssh", a=0.2022, b=0.1793, r=0.0534, wall="A", Nside=40)
    ql = dict(kind="quan_loncar", period=0.4616, radii=[0.0969, 0.1419], n_taper=2, n_mirror=25)
    rows = [_nb(d, 29, 2.8e5, 2.4e4, 1.547, name="ssh_A[Nside=40]"),          # mirror-limited
            _nb(ql, 29, 3.6e3, 4.7e6, 1.5385, name="quan_loncar[n_mirror=25]")]  # radiation-limited
    g = P.gonogo("E3", rows)
    lim = {v["design"]: v["limited_by"] for v in g["variants"].values()}
    assert lim == {"ssh_A": "mirror", "quan_loncar": "radiation"}
    assert g["comparison"]["comparable"] is False and g["go"] is False


def test_dispatch_runs_small_jobs_locally_first(monkeypatch, capsys):
    modal = pytest.importorskip("modal")  # app.py needs the client library, not a connection
    import app as A
    calls = {}
    monkeypatch.setattr(A, "_existing", lambda exp: set())
    monkeypatch.setattr(A, "load_results", lambda exp: _fake_e2_bighole() if exp == "E2" else [])
    monkeypatch.setattr(A, "_spent_usd", lambda: 7.62)
    def fake_local(small, local_np, dry_run):
        calls["small"] = [j["kernel"] for j in small]
        return []
    monkeypatch.setattr(A, "_run_small_locally", fake_local)
    A._dispatch("E3", "", dry_run=True)
    out = capsys.readouterr().out
    assert set(calls["small"]) == {"mpb_cell", "ql_recipe"}  # MPB jobs stay local; 3D nanobeams go to Modal
    assert "[budget] spent $7.62" in out
    # no local env and no --modal-small: small jobs are held back, not sent to Modal
    monkeypatch.setattr(A, "_run_small_locally", lambda small, n, d: None)
    A._dispatch("E3", "", dry_run=True)
    assert "Not sending them to Modal" in capsys.readouterr().out


@pytest.mark.skipif(not os.path.isdir(SNAPSHOT_E3), reason="Modal results snapshot not present")
def test_e3_snapshot_flags_mixed_channels():
    import glob
    import json
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(SNAPSHOT_E3, "**", "*.json"), recursive=True)]
    g = P.gonogo("E3", rows)
    if not any(v["design"] == "quan_loncar" and 29 in v["by_res"] for v in g["variants"].values()):
        pytest.skip("snapshot predates the res-29 tuned control")
    # SSH walls mirror-limited at Nside 40 vs the radiation-limited tuned control (Q_axial/Q_rad ~ 1300)
    assert g["comparison"]["limited_by"]["quan_loncar"] == "radiation"
    assert g["comparison"]["limited_by"]["ssh_A"] == "mirror" and g["comparison"]["comparable"] is False
