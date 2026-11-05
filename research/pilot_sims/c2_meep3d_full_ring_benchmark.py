"""
Benchmark (c-ii): direct 3D MEEP timing + memory of the FULL repo ring (R=10 um, w=0.5,
t=0.4, oxide-clad SiN, 250 air holes with SSH spacing), Mirror(Z) symmetry only.
Usage: [mpirun -np N] python c2_meep3d_full_ring_benchmark.py res T_bench [R w t]
"""
import sys, time, json, resource
import numpy as np
import meep as mp

res = int(sys.argv[1]); T_bench = float(sys.argv[2])
R = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
w = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
t = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4
a, b, r = 0.35, 0.15, 0.05
dpml = 1.0
L = a + b; N = int(2 * np.pi * R / L); sc = (2 * np.pi * R / N) / L
ang, th = 0.0, []
for i in range(N):
    th.append(ang); ang += a * sc / R
    th.append(ang); ang += b * sc / R
sxy = 2 * (R + w / 2 + 1.0) + 2 * dpml
sz = t + 2 * 0.8 + 2 * dpml
ox = mp.Medium(index=1.46)
geo = [mp.Cylinder(radius=R + w / 2, height=t, material=mp.Medium(index=2.0)),
       mp.Cylinder(radius=R - w / 2, height=t, material=ox)]
geo += [mp.Cylinder(radius=r, height=t, center=mp.Vector3(R * np.cos(q), R * np.sin(q), 0), material=mp.air) for q in th]
src = [mp.Source(mp.GaussianSource(1 / 1.547, fwidth=0.1), component=mp.Ey, center=mp.Vector3(R, 0.05, 0))]
sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy, sz), geometry=geo, sources=src, resolution=res,
                    boundary_layers=[mp.PML(dpml)], default_material=ox, symmetries=[mp.Mirror(mp.Z, phase=+1)])
t0 = time.time(); sim.init_sim(); t_init = time.time() - t0
t0 = time.time(); sim.run(until=T_bench); t_run = time.time() - t0
nsteps = T_bench / (0.5 / res)
rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
tot_rss = mp.sum_to_all(rss_gb) if hasattr(mp, "sum_to_all") else rss_gb * mp.count_processors()
nvox = sxy * sxy * sz * res ** 3
if mp.am_master():
    out = dict(res=res, R=R, w=w, t=t, nvox_full=nvox, nproc=mp.count_processors(), t_init=t_init,
               sec_per_step=t_run / nsteps, sec_per_time_unit=t_run / T_bench, peak_rss_total_GB=tot_rss)
    f = 1 / 1.547
    for Q in (1e4, 1e5, 1e6):
        out[f"hours_T=tau_Q{Q:.0e}"] = Q / (2 * np.pi * f) * t_run / T_bench / 3600
    print(f"BENCHRING3D res={res} vox={nvox:.3g} np={mp.count_processors()} init={t_init:.1f}s "
          f"{1e3*t_run/nsteps:.0f} ms/step {t_run/T_bench:.2f} s/time-unit peakRSS~{tot_rss:.2f} GB")
    print("RESULT " + json.dumps(out))
