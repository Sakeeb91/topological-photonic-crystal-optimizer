"""
Benchmark (a): MPB 3D band structure of the repo's SSH unit cell.

Unit cell (length units: um, MPB frequency f = 1/lambda[um]):
  SiN strip (n=2.0), thickness t, width w, on SiO2 substrate (n=1.46), air top.
  Two air holes (radius r, through the SiN) per period L = a + b, center-to-center
  spacing a inside the cell and b across the cell boundary.
  Supercell Ly x Lz transverse.  Only y-odd (TE-like, Ey-dominant) modes computed
  (mirror plane y=0 exists; z-symmetry broken by substrate).

Usage: python a_mpb_ssh_unitcell.py a b r w t res [nk] [clad]
  clad = 'sub' (SiO2 below, air above; default) or 'oxide' (SiO2 everywhere, air holes)
"""
import sys, time, json
import numpy as np
import meep as mp
from meep import mpb

a, b, r, w, t, res = map(float, sys.argv[1:7])
nk = int(sys.argv[7]) if len(sys.argv) > 7 else 1
clad = sys.argv[8] if len(sys.argv) > 8 else "sub"
res = int(res)
L = a + b
Ly, Lz = (3.0, 3.0) if clad != "sub" else (3.0, 4.0)
nSiN, nOx = 2.0, 1.46

geometry = []
hole_mat = mp.air
if clad in ("oxide", "oxide_air"):      # SiO2 everywhere, air holes
    default_material = mp.Medium(index=nOx)
elif clad == "oxide_fill":              # SiO2 everywhere, holes back-filled with SiO2
    default_material = mp.Medium(index=nOx); hole_mat = mp.Medium(index=nOx)
elif clad == "susp":                    # suspended SiN, air everywhere
    default_material = mp.air
else:
    default_material = mp.air
    geometry.append(mp.Block(center=mp.Vector3(0, 0, -t / 2 - Lz / 4),
                             size=mp.Vector3(mp.inf, mp.inf, Lz / 2),
                             material=mp.Medium(index=nOx)))
geometry.append(mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t),
                         material=mp.Medium(index=nSiN)))
for x0 in (-a / 2, a / 2):
    if r > 0:
        geometry.append(mp.Cylinder(radius=r, height=t, center=mp.Vector3(x0, 0, 0),
                                    material=hole_mat))

lattice = mp.Lattice(size=mp.Vector3(L, Ly, Lz))
# k in units of reciprocal lattice vectors (kx=0.5 -> zone edge pi/L)
if nk == 1:
    k_points = [mp.Vector3(0.5)]
else:
    k_points = mp.interpolate(nk - 2, [mp.Vector3(0.3), mp.Vector3(0.5)])

num_bands = 4
ms = mpb.ModeSolver(geometry_lattice=lattice, geometry=geometry,
                    default_material=default_material, k_points=k_points,
                    resolution=res, num_bands=num_bands, tolerance=1e-7)
t0 = time.time()
ms.run_yodd()
wall = time.time() - t0
freqs = np.array(ms.all_freqs)  # shape (nk, nbands), units 1/um
kx = np.array([k.x for k in k_points])
k_abs = kx / L  # 1/um
n_lc = nOx if True else 1.0
f_ll = k_abs / (1.0 if clad == "susp" else nOx)  # substrate (or oxide) light line
out = dict(a=a, b=b, r=r, w=w, t=t, res=res, clad=clad, L=L, wall_s=wall,
           kx=kx.tolist(), freqs=freqs.tolist(), f_lightline=f_ll.tolist())
fe = freqs[-1]  # zone edge
f1, f2 = fe[0], fe[1]
gap = f2 - f1
mid = 0.5 * (f1 + f2)
out.update(edge_f1=f1, edge_f2=f2, gap_abs=gap, gap_mid_gap_ratio=gap / mid,
           lambda1_um=1 / f1, lambda2_um=1 / f2, lambda_mid_um=1 / mid,
           dlambda_nm=1e3 * (1 / f1 - 1 / f2), f_ll_edge=f_ll[-1],
           below_lightline=bool(f2 < f_ll[-1]))
print("RESULT " + json.dumps(out))
print(f"a={a} b={b} r={r} w={w} t={t} res={res} clad={clad}: zone-edge TE bands "
      f"f1={f1:.5f} f2={f2:.5f} (1/um) -> lambda {1/f1:.4f}/{1/f2:.4f} um, "
      f"gap {1e3*(1/f1-1/f2):.2f} nm, gap/mid={gap/mid*100:.3f}%, "
      f"normalized L/lambda mid={mid*L:.4f}, light line f={f_ll[-1]:.4f}, wall {wall:.1f}s")
