"""
Stretch (d, cross-check): legume guided-mode expansion of a suspended SiN SSH nanobeam
supercell with two domain walls (type-B at x=0, type-A at the supercell edge).
Gives frequencies and radiative Q (perturbative, Im(omega) from coupling to radiation).
Caveat: GME uses a homogeneous-slab basis (here eps_eff of the strip); for a narrow
strip in air this is an approximation.  Use MEEP 3D as reference.

Usage: python d2_legume_gme_domainwall.py a b r w t Nside gmax Ly
"""
import sys, time, json
import numpy as np
import legume

a, b, r, w, t = map(float, sys.argv[1:6]); Nside = int(sys.argv[6]); gmax = float(sys.argv[7]); Ly = float(sys.argv[8])
# right half gaps outward from centre hole: b, a, b, a ... (type-B wall at x=0)
xs = [0.0]; x = 0.0
for i in range(2 * Nside):
    x += (b, a)[i % 2]; xs += [x, -x]
xmax = x
# close the supercell: last gap used was 'a' (i odd) -> add gap b so the edge wall is type A-like
Lx = 2 * xmax + (b if (2 * Nside - 1) % 2 == 1 else a)
lat = legume.Lattice([Lx, 0], [0, Ly])
phc = legume.PhotCryst(lat, eps_l=1.0, eps_u=1.0)
phc.add_layer(d=t, eps_b=1.0)
phc.add_shape(legume.Poly(eps=4.0, x_edges=[-Lx / 2, Lx / 2, Lx / 2, -Lx / 2],
                          y_edges=[-w / 2, -w / 2, w / 2, w / 2]))
for xh in xs:
    phc.add_shape(legume.Circle(eps=1.0, x_cent=xh, y_cent=0.0, r=r))
gme = legume.GuidedModeExp(phc, gmax=gmax)
t0 = time.time()
gme.run(kpoints=np.array([[0.0], [0.0]]), gmode_inds=[0], numeig=24, eig_solver="eigsh",
        eig_sigma=0.85, compute_im=True, verbose=False, eps_eff="average")
wall = time.time() - t0
f = gme.freqs[0]; fi = gme.freqs_im[0]
Q = f / (2 * np.maximum(fi, 1e-30))
print(f"LEGUME Lx={Lx:.3f} Ly={Ly} gmax={gmax} nG={gme.gvec.shape[1]} wall={wall:.1f}s")
for fk, qk in zip(f, Q):
    print(f"  GME f={fk:.5f} lam={1/fk:.4f} Q_rad={qk:.4g}")
json.dump(dict(Lx=Lx, Ly=Ly, gmax=gmax, nG=int(gme.gvec.shape[1]), wall=wall, f=f.tolist(), Q=Q.tolist()),
          open(f"d2_legume_r{r}_N{Nside}_g{gmax}.json", "w"))
