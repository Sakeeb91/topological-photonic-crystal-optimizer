"""
Benchmark (b): 2D MEEP time-per-step for the repo's full SSH ring.
Geometry exactly as repo _create_meep_geometry comments: background SiO2 n=1.46,
ring n=2.0 (R, w), air holes on the centerline with SSH spacing a/b wrapped
uniformly (no domain wall).  2D TE (Hz) polarisation.

Usage: python b_meep2d_ring_benchmark.py res nsteps_time [harminv_T] [n_ring]
  nsteps_time : MEEP time units to run for timing (e.g. 20)
  harminv_T   : optional: if >0 run a Gaussian-pulse + Harminv for this many time
                units after the source, report modes (Q, lambda).
  n_ring      : ring index (default 2.0 as in repo; 1.70 ~ slab effective index)
"""
import sys, time, json
import numpy as np
import meep as mp

res = int(sys.argv[1])
T_bench = float(sys.argv[2])
T_harm = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
n_ring = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
a, b, R, w = 0.35, 0.15, 10.0, 0.5
r = float(sys.argv[5]) if len(sys.argv) > 5 else 0.05
dpml = 1.0
lam0 = 1.547

# repo hole placement (no disorder)
L = a + b
N = int(2 * np.pi * R / L)
scale = (2 * np.pi * R / N) / L
ang, holes = 0.0, []
for i in range(N):
    holes.append(ang); ang += a * scale / R
    holes.append(ang); ang += b * scale / R

sx = 2 * (R + w / 2 + 1.0) + 2 * dpml
cell = mp.Vector3(sx, sx, 0)
geo = [mp.Cylinder(radius=R + w / 2, material=mp.Medium(index=n_ring)),
       mp.Cylinder(radius=R - w / 2, material=mp.Medium(index=1.46))]
geo += [] if r <= 0 else [mp.Cylinder(radius=r, center=mp.Vector3(R * np.cos(t), R * np.sin(t)), material=mp.air) for t in holes]

fcen, df = 1 / lam0, 0.15 / lam0
src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Hz,
                 center=mp.Vector3(R + 0.1, 0.03))]
sim = mp.Simulation(cell_size=cell, geometry=geo, sources=src, resolution=res,
                    boundary_layers=[mp.PML(dpml)], default_material=mp.Medium(index=1.46))
t0 = time.time(); sim.init_sim(); t_init = time.time() - t0
dt = sim.fields.dt if hasattr(sim.fields, 'dt') else 0.5 / res
t0 = time.time(); sim.run(until=T_bench); t_run = time.time() - t0
nsteps = int(round(T_bench / (0.5 / res)))
sps = t_run / nsteps
npix = (sx * res) ** 2
out = dict(res=res, n_ring=n_ring, r=r, n_holes=len(holes), cell_um=sx, npix=npix, t_init=t_init,
           sec_per_step=sps, sec_per_time_unit=t_run / T_bench, ns_per_pixel_step=1e9 * sps / npix)
# extrapolation: time units needed ~ Q/(2 pi f) * alpha with alpha~1 (one photon lifetime),
# typical Harminv practice alpha~0.3-1 for well-isolated modes
for Q in (1e4, 1e5, 1e6):
    tau = Q / (2 * np.pi * fcen)
    out[f"hours_for_T=tau_Q{Q:.0e}"] = tau * t_run / T_bench / 3600
print(f"BENCH2D res={res} pix={npix:.3g} init={t_init:.1f}s  {sps*1e3:.2f} ms/step  "
      f"{t_run/T_bench:.2f} s per MEEP time unit  ({1e9*sps/npix:.1f} ns/pixel/step)")
if T_harm > 0:
    h = mp.Harminv(mp.Hz, mp.Vector3(R - 0.1, 0.0), fcen, df)
    t0 = time.time()
    sim.run(mp.after_sources(h), until_after_sources=T_harm)
    out["harminv_wall_s"] = time.time() - t0
    out["T_harm"] = T_harm
    modes = sorted([(m.freq, m.Q, abs(m.amp)) for m in h.modes], key=lambda x: -x[2])
    out["modes"] = [dict(lam_um=1 / f, Q=q, amp=A) for f, q, A in modes[:12]]
    for f, q, A in modes[:12]:
        print(f"  MODE lam={1/f:.5f} um  Q={q:.4g}  |amp|={A:.3g}")
print("RESULT " + json.dumps(out))
