"""
Geometry, fabrication rules and disorder for the Modal experiments.

Pure numpy; no MEEP import, so it runs (and is unit-tested) anywhere.
Length unit: um. Disorder sigmas are given in nm and converted here.

Hole lists are lists of dicts {x, y, r, ord}. `ord` is a signed ordinal
(0 = centre hole, +i = i-th hole at x > 0, -i = i-th hole at x < 0) that keys
the common-random-number (CRN) streams, so hole k gets the same draw in every
design for a given (seed, realization).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Platforms (docs/LITERATURE_REVIEW.md section 3.7). Indices at 1.55 um.
# --------------------------------------------------------------------------
PLATFORMS: dict[str, dict[str, Any]] = {
    # ANT NanoSOI: 220 nm Si on buried oxide; 60 nm minimum feature / gap.
    "si220": dict(n_core=3.476, n_ox=1.444, t=0.22, min_feature=0.060, min_web=0.060),
    # ANT SiN MPW: 400 nm SiN on oxide, 100 keV EBL; 120 nm minimum feature / gap.
    # n_core=2.0 and n_ox=1.46 match the pilot scripts.
    "sin400": dict(n_core=2.0, n_ox=1.46, t=0.40, min_feature=0.120, min_web=0.120),
}

# Cladding conventions (shared by every kernel):
#   oxide       oxide above and below, air holes
#   oxide_fill  oxide above and below, holes filled with oxide (ANT SiN cladding fills holes)
#   sub         oxide below, air above, air holes
#   susp        air everywhere (suspended membrane)
CLADS = ("oxide", "oxide_fill", "sub", "susp")

KMAX = 4096  # CRN stream half-length; supports up to 4096 holes on each side


def platform(name: str, **override) -> dict[str, Any]:
    if name not in PLATFORMS:
        raise ValueError(f"unknown platform {name!r}; choose from {sorted(PLATFORMS)}")
    p = dict(PLATFORMS[name]); p.update({k: v for k, v in override.items() if v is not None})
    p["name"] = name
    return p


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def _ordinals(xs: list[float]) -> list[int]:
    pos = sorted([x for x in xs if x > 1e-12])
    neg = sorted([x for x in xs if x < -1e-12], reverse=True)
    rank = {}
    for i, x in enumerate(pos):
        rank[("p", round(x, 9))] = i + 1
    for i, x in enumerate(neg):
        rank[("n", round(x, 9))] = -(i + 1)
    out = []
    for x in xs:
        if abs(x) <= 1e-12:
            out.append(0)
        elif x > 0:
            out.append(rank[("p", round(x, 9))])
        else:
            out.append(rank[("n", round(x, 9))])
    return out


def _holes(xs, rs) -> list[dict]:
    ords = _ordinals(list(xs))
    hs = [dict(x=float(x), y=0.0, r=float(r), ord=int(o)) for x, r, o in zip(xs, rs, ords)]
    return sorted(hs, key=lambda h: h["x"])


def ssh_nanobeam(a: float, b: float, r: float, Nside: int, wall: str = "A", M: float = 0.0) -> list[dict]:
    """Straight SSH chain, same conventions as research/pilot_sims/c_meep3d_nanobeam.py and
    d3_meep3d_tapered_wall.py. a, b are centre-to-centre hole spacings.

    wall='none' : uniform a,b,a,b chain (no domain wall)
    wall='A'    : gaps outward from the centre hole a, b, a, b, ...
    wall='B'    : gaps outward from the centre hole b, a, b, a, ...
    wall='tanh' : type-B wall with dimerization Delta_i = Delta0 * tanh((i + 0.5) / M)
    """
    xs: list[float] = []
    if wall == "none":
        x = -Nside * (a + b) + a / 2
        for _ in range(2 * Nside):
            xs += [x, x + a]; x += a + b
    elif wall in ("A", "B"):
        g = (b, a) if wall == "B" else (a, b)
        xs = [0.0]; x = 0.0
        for i in range(2 * Nside):
            x += g[i % 2]; xs += [x, -x]
    elif wall == "tanh":
        if M <= 0:
            raise ValueError("tanh wall needs M > 0")
        L = a + b; D0 = (a - b) / 2
        xs = [0.0]; x = 0.0
        for i in range(Nside):
            D = D0 * math.tanh((i + 0.5) / M)
            for gap in (L / 2 - D, L / 2 + D):
                x += gap; xs += [x, -x]
    else:
        raise ValueError(f"unknown wall {wall!r}")
    return _holes(xs, [r] * len(xs))


def quan_loncar_nanobeam(period: float, r_center: float, r_end: float, n_taper: int, n_mirror: int) -> list[dict]:
    """Trivial Gaussian-mirror nanobeam (Quan & Loncar, Opt. Express 19, 18529 (2011)).

    Constant period, zero-length cavity at x = 0 (holes at +-(i + 1/2) period).
    The filling fraction f = pi r^2 / (period * w) is tapered quadratically from the
    centre value to the mirror value over n_taper holes per side, which makes the
    mirror strength grow about linearly and the field envelope about Gaussian.
    Then n_mirror more holes per side at r_end.
    """
    fc, fe = r_center ** 2, r_end ** 2  # f is proportional to r^2 at fixed period and w
    rs = []
    for i in range(n_taper):
        s = (i / max(n_taper - 1, 1)) ** 2 if n_taper > 1 else 1.0
        rs.append(math.sqrt(fc + (fe - fc) * s))
    rs += [r_end] * n_mirror
    xs, rr = [], []
    for i, ri in enumerate(rs):
        x = (i + 0.5) * period
        xs += [x, -x]; rr += [ri, ri]
    return _holes(xs, rr)


def ring_gap_sequence(a: float, b: float, N1: int, N2: int) -> list[float]:
    """Gap list for a closed two-segment SSH ring/supercell: N1 cells of (a, b) then N2
    cells of (b, a). The junctions give one 'b b' and one 'a a' wall, i.e. one A-type
    and one B-type wall (a two-segment ring always has one of each)."""
    return [a, b] * N1 + [b, a] * N2


def periodic_supercell(a: float, b: float, r: float, N1: int, N2: int) -> tuple[list[dict], float, dict]:
    """Unrolled ring: holes on a line of length C = (N1 + N2)(a + b), periodic in x,
    centred so the cell spans [-C/2, C/2). Returns (holes, C, wall_x)."""
    gaps = ring_gap_sequence(a, b, N1, N2)
    C = float(sum(gaps))
    xs = [0.0]
    for g in gaps[:-1]:
        xs.append(xs[-1] + g)
    # Hole 2*N1 has gap b on both sides: a type-B wall (pilot convention, wall='B').
    # Hole 0 has gap a on both sides (wrap-around): a type-A wall.
    x_B = xs[2 * N1]; x_A = 0.0
    shift = -(x_A + x_B) / 2  # centre the two walls in the cell, away from the boundary
    xs = [((x + shift + C / 2) % C) - C / 2 for x in xs]
    holes = sorted([dict(x=x, y=0.0, r=r, ord=i) for i, x in enumerate(xs)], key=lambda h: h["x"])
    walls = {"A": x_A + shift, "B": x_B + shift}
    return holes, C, walls


# --------------------------------------------------------------------------
# Fabrication rules
# --------------------------------------------------------------------------
def fab_check(holes: list[dict], w: float, plat: dict, periodic_length: float | None = None) -> dict:
    """Check minimum feature (hole diameter, rail width) and web (gap minus hole diameter,
    generalised to unequal radii: dx - r_i - r_j) against a platform's rules."""
    mf, mw = plat["min_feature"], plat["min_web"]
    hs = sorted(holes, key=lambda h: h["x"])
    viol = []
    min_d = min((2 * h["r"] for h in hs), default=float("inf"))
    if min_d < mf - 1e-12:
        viol.append(f"hole diameter {1e3*min_d:.1f} nm < min feature {1e3*mf:.0f} nm")
    webs = [hs[i + 1]["x"] - hs[i]["x"] - hs[i]["r"] - hs[i + 1]["r"] for i in range(len(hs) - 1)]
    if periodic_length is not None and len(hs) > 1:
        webs.append(hs[0]["x"] + periodic_length - hs[-1]["x"] - hs[0]["r"] - hs[-1]["r"])
    min_web = min(webs, default=float("inf"))
    if min_web < mw - 1e-12:
        n_bad = sum(1 for v in webs if v < mw - 1e-12)
        viol.append(f"web {1e3*min_web:.1f} nm < min web {1e3*mw:.0f} nm ({n_bad} of {len(webs)} webs)")
    rails = [w / 2 - abs(h["y"]) - h["r"] for h in hs]
    min_rail = min(rails, default=float("inf"))
    if min_rail < max(mf, mw) - 1e-12:
        viol.append(f"side rail {1e3*min_rail:.1f} nm < {1e3*max(mf, mw):.0f} nm")
    return dict(ok=not viol, violations=viol, platform=plat["name"],
                min_hole_diameter_nm=1e3 * min_d, min_web_nm=1e3 * min_web, min_rail_nm=1e3 * min_rail)


def ssh_cell_fab_check(a: float, b: float, r: float, w: float, plat: dict) -> dict:
    """Unit-cell check: two webs (a - 2r, b - 2r), diameter 2r, rails (w - 2r)/2."""
    L = a + b
    holes = [dict(x=-a / 2, y=0.0, r=r, ord=0), dict(x=a / 2, y=0.0, r=r, ord=1)]
    return fab_check(holes, w, plat, periodic_length=L)


# --------------------------------------------------------------------------
# Disorder with common random numbers
# --------------------------------------------------------------------------
DISORDER_CLASSES = {"radius": 1, "position_x": 2, "offset_y": 3, "cd_bias": 4}


def _stream(seed: int, realization: int, cls: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), int(realization), DISORDER_CLASSES[cls]])


def apply_disorder(holes: list[dict], spec: dict | None) -> list[dict]:
    """spec = {seed, realization, sigma_r_nm, sigma_x_nm, sigma_y_nm, sigma_bias_nm}.
    Each class draws from its own stream, indexed by hole ordinal, so switching a class
    on or off, or changing the design, leaves the other draws unchanged (CRN)."""
    if not spec:
        return [dict(h) for h in holes]
    seed, k = int(spec.get("seed", 0)), int(spec.get("realization", 0))
    sr = 1e-3 * float(spec.get("sigma_r_nm", 0.0))
    sx = 1e-3 * float(spec.get("sigma_x_nm", 0.0))
    sy = 1e-3 * float(spec.get("sigma_y_nm", 0.0))
    sb = 1e-3 * float(spec.get("sigma_bias_nm", 0.0))
    dr = _stream(seed, k, "radius").standard_normal(2 * KMAX + 1)
    dx = _stream(seed, k, "position_x").standard_normal(2 * KMAX + 1)
    dy = _stream(seed, k, "offset_y").standard_normal(2 * KMAX + 1)
    bias = float(_stream(seed, k, "cd_bias").standard_normal())
    out = []
    for h in holes:
        i = h["ord"] + KMAX
        if not 0 <= i < 2 * KMAX + 1:
            raise ValueError("hole ordinal outside CRN stream")
        out.append(dict(h, r=max(h["r"] + sr * dr[i] + sb * bias, 1e-4),
                        x=h["x"] + sx * dx[i], y=h["y"] + sy * dy[i]))
    return out


def breaks_y_mirror(spec: dict | None) -> bool:
    return bool(spec) and float(spec.get("sigma_y_nm", 0.0)) > 0


# --------------------------------------------------------------------------
# Design dispatch
# --------------------------------------------------------------------------
def build_design(d: dict) -> list[dict]:
    kind = d.get("kind", "ssh")
    if kind == "ssh":
        return ssh_nanobeam(d["a"], d["b"], d["r"], int(d["Nside"]), d.get("wall", "A"), float(d.get("M", 0.0)))
    if kind == "quan_loncar":
        return quan_loncar_nanobeam(d["period"], d["r_center"], d["r_end"], int(d["n_taper"]), int(d["n_mirror"]))
    raise ValueError(f"unknown design kind {kind!r}")
