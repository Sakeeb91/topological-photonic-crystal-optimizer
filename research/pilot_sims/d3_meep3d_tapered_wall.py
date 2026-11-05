"""
Stretch (d3): tapered SSH domain wall (wall=T<M>), otherwise identical to c_meep3d_nanobeam.py.
Benchmarks (c) and (d): 3D MEEP straight SSH nanobeam (SiN strip with air holes).

Structure: SiN (n=2.0) strip of width w, thickness t along x, through-holes radius r.
Hole sequence is mirror-symmetric about x=0 (a hole sits at x=0) when wall != 'none':
  wall='B'   : gaps outward from the centre hole are b, a, b, a, ...  (left side mirrored)
  wall='A'   : gaps outward from the centre hole are a, b, a, b, ...
  wall='none': uniform a,b,a,b,... chain (no domain wall)  -> timing only
The strip runs into the PML at both ends (so Q includes leakage through the finite mirrors).
Cladding: 'susp' (air everywhere) or 'oxide' (SiO2 everywhere, air holes).
Symmetries: Mirror(Y, phase=-1) (TE0-like, Ey even) and, for symmetric claddings,
Mirror(Z, phase=+1) (TE-like).  Excitation: Ey Gaussian dipole slightly off x=0 so both
x-parities are excited.

Usage: [mpirun -np 4] python c_meep3d_nanobeam.py clad a b r w t Nside wall res T_bench T_harm fcen df
"""
import sys, time, json
import numpy as np
import meep as mp

clad, = sys.argv[1:2]
a, b, r, w, t = map(float, sys.argv[2:7])
Nside = int(sys.argv[7]); wall = sys.argv[8]; res = int(sys.argv[9])
T_bench = float(sys.argv[10]); T_harm = float(sys.argv[11])
fcen = float(sys.argv[12]); df = float(sys.argv[13])
dpml = 1.0
pad_y, pad_z = 1.0, 0.8
nSiN, nOx = 2.0, 1.46
bg = mp.air if clad == "susp" else mp.Medium(index=nOx)

# hole x-positions
xs = []
if wall == "none":
    x = -Nside * (a + b) + a / 2
    for i in range(2 * Nside):
        xs += [x, x + a]; x += a + b
elif wall.startswith("T"):
    # Tapered (Jackiw-Rebbi-like) type-B wall: dimerization Delta_i = Delta0*tanh((i+0.5)/M)
    # for the i-th unit cell outward from the centre hole; gaps (L/2-Delta_i, L/2+Delta_i).
    M = float(wall[1:]); L = a + b; D0 = (a - b) / 2
    xs = [0.0]; x = 0.0
    for i in range(Nside):
        D = D0 * np.tanh((i + 0.5) / M) if M > 0 else D0
        for gap in (L / 2 - D, L / 2 + D):
            x += gap; xs += [x, -x]
else:
    g = (b, a) if wall == "B" else (a, b)
    xs = [0.0]; x = 0.0
    for i in range(2 * Nside):
        x += g[i % 2]; xs += [x, -x]
xmax = max(abs(v) for v in xs)
sx = 2 * (xmax + 1.0) + 2 * dpml
sy = w + 2 * pad_y + 2 * dpml
sz = t + 2 * pad_z + 2 * dpml
cell = mp.Vector3(sx, sy, sz)
geo = [mp.Block(center=mp.Vector3(), size=mp.Vector3(mp.inf, w, t), material=mp.Medium(index=nSiN))]
geo += [mp.Cylinder(radius=r, height=t, center=mp.Vector3(xh, 0, 0), material=mp.air) for xh in xs]
src = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Ey, center=mp.Vector3(0.07, 0, 0))]
sym = [mp.Mirror(mp.Y, phase=-1)]
if clad in ("susp", "oxide"):
    sym.append(mp.Mirror(mp.Z, phase=+1))
sim = mp.Simulation(cell_size=cell, geometry=geo, sources=src, resolution=res, symmetries=sym,
                    boundary_layers=[mp.PML(dpml)], default_material=bg)
t0 = time.time(); sim.init_sim(); t_init = time.time() - t0
t0 = time.time(); sim.run(until=T_bench); t_run = time.time() - t0
nsteps = T_bench / (0.5 / res)
nvox = sx * sy * sz * res ** 3
out = dict(clad=clad, a=a, b=b, r=r, w=w, t=t, Nside=Nside, wall=wall, res=res, nholes=len(xs),
           cell=[sx, sy, sz], nvox_full=nvox, nproc=mp.count_processors(), t_init=t_init,
           sec_per_step=t_run / nsteps, sec_per_time_unit=t_run / T_bench,
           ns_per_voxel_step_full=1e9 * t_run / nsteps / nvox)
if mp.am_master():
    print(f"BENCH3D clad={clad} wall={wall} N={Nside} res={res} vox(full)={nvox:.3g} np={mp.count_processors()} "
          f"init={t_init:.1f}s {1e3*t_run/nsteps:.1f} ms/step {t_run/T_bench:.2f} s/time-unit")
if T_harm > 0:
    h = mp.Harminv(mp.Ey, mp.Vector3(0.07, 0, 0), fcen, df)
    h2 = mp.Harminv(mp.Ey, mp.Vector3(-0.21, 0, 0), fcen, df)
    t0 = time.time()
    sim.run(mp.after_sources(h, h2), until_after_sources=T_harm)
    out["harminv_wall_s"] = time.time() - t0; out["T_harm"] = T_harm
    modes = sorted({(round(m.freq, 6), m.Q, abs(m.amp), m.err) for m in h.modes + h2.modes}, key=lambda x: -x[2])
    out["modes"] = [dict(f=f, lam_um=1 / f, Q=q, amp=A, err=e) for f, q, A, e in modes[:10]]
    if mp.am_master():
        for f, q, A, e in modes[:10]:
            print(f"  MODE f={f:.5f} lam={1/f:.5f}um Q={q:.4g} |amp|={A:.3g} err={e:.2g}")
if mp.am_master():
    print("RESULT " + json.dumps(out))
