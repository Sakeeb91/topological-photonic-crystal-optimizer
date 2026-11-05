"""
Extra benchmark (e): exact 3D bend-radiation Q of the BARE ring (no holes) using MEEP
cylindrical coordinates (axisymmetric, angular order m) -> a 2D (r,z) simulation.
This is the upper bound on Q from bending loss alone for the repo cross-section;
holes can only add loss (or, at the SSH band edge, modify group index).

Usage: python e_meep_cyl_bare_ring_Q.py R w t clad m res T_after
  clad: 'oxide' (SiO2 everywhere) | 'airtop' (SiO2 substrate below, air above)
"""
import sys, json, time
import numpy as np
import meep as mp

R, w, t = map(float, sys.argv[1:4]); clad = sys.argv[4]; m = int(sys.argv[5])
res = int(sys.argv[6]); T_after = float(sys.argv[7])
nSiN, nOx = 2.0, 1.46
dpml, padr, padz = 1.0, 2.5, 1.5
sr = R + w / 2 + padr + dpml
sz = t + 2 * padz + 2 * dpml
cell = mp.Vector3(sr, 0, sz)
geo = []
if clad == "airtop":
    bg = mp.air
    geo.append(mp.Block(center=mp.Vector3(sr / 2, 0, -t / 2 - sz / 4), size=mp.Vector3(sr, mp.inf, sz / 2),
                        material=mp.Medium(index=nOx)))
else:
    bg = mp.Medium(index=nOx)
geo.append(mp.Block(center=mp.Vector3(R, 0, 0), size=mp.Vector3(w, mp.inf, t), material=mp.Medium(index=nSiN)))
lam0 = 1.547
fcen, df = 1 / lam0, 0.25 / lam0
src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Er, center=mp.Vector3(R + 0.05, 0, 0.03))]
sim = mp.Simulation(cell_size=cell, geometry=geo, boundary_layers=[mp.PML(dpml)], resolution=res,
                    sources=src, dimensions=mp.CYLINDRICAL, m=m, default_material=bg)
h = mp.Harminv(mp.Er, mp.Vector3(R + 0.05, 0, 0.03), fcen, df)
t0 = time.time()
sim.run(mp.after_sources(h), until_after_sources=T_after)
wall = time.time() - t0
modes = sorted([(md.freq, md.Q, abs(md.amp), md.err) for md in h.modes], key=lambda x: -x[2])
out = dict(R=R, w=w, t=t, clad=clad, m=m, res=res, T_after=T_after, wall_s=wall,
           modes=[dict(lam_um=1 / f, Q=q, amp=A, err=e) for f, q, A, e in modes[:6]])
for f, q, A, e in modes[:6]:
    print(f"CYL R={R} w={w} t={t} {clad} m={m}: lam={1/f:.4f} um Q={q:.4g} |amp|={A:.3g} err={e:.1g}")
print("RESULT " + json.dumps(out))
