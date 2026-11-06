"""
Is the repo's nominal waveguide even guided?  MPB cross-section (no holes) TE0 effective
index at lambda = 1.547 um for SiN (n=2.0) strips on SiO2 (n=1.46), air-top or oxide-clad.
Uses MPB find_k on a 2D (y,z) cross-section with propagation along x.
"""
import sys, json
import numpy as np
import meep as mp
from meep import mpb

lam = 1.547
f0 = 1 / lam
Ly, Lz = 4.0, 4.0
res = 40
cases = [(0.5, 0.4), (0.5, 0.8), (0.7, 0.4), (1.0, 0.4), (1.2, 0.4), (1.0, 0.8), (1.6, 0.8)]
results = []
for clad in ("air_top", "oxide"):
    for w, t in cases:
        geo = []
        if clad == "air_top":
            dm = mp.air
            geo.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - Lz / 4), size=mp.Vector3(mp.inf, mp.inf, Lz / 2),
                                material=mp.Medium(index=1.46)))
        else:
            dm = mp.Medium(index=1.46)
        geo.append(mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t), material=mp.Medium(index=2.0)))
        ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0, Ly, Lz)), geometry=geo,
                            default_material=dm, resolution=res, num_bands=1)
        try:
            k = ms.find_k(mp.ODD_Y, f0, 1, 1, mp.Vector3(1, 0, 0), 1e-4, f0 * 1.6, f0 * 1.0, f0 * 2.0)
            neff = k[0] / f0
        except Exception as e:  # noqa
            neff = float("nan")
        guided = neff > 1.46 + 0.005
        results.append(dict(clad=clad, w=w, t=t, neff_TE0=neff, guided=bool(guided)))
        print(f"NEFF clad={clad:8s} w={w:.2f} t={t:.2f}  n_eff(TE0, 1.547um) = {neff:.4f}  "
              f"{'GUIDED' if guided else 'NOT guided / at cutoff (n_eff<=n_SiO2)'}")
json.dump(results, open("a1_neff.json", "w"), indent=1)
