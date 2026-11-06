"""2D MPB (TE, in-plane E) check of the Hotte-Kilburn & Bianucci (J. Opt. 2024) designs.
Strip n=1.81, w=0.465 um, two air holes per cell, background n_bg. For each geometry convention,
print the zone-edge (X) bands below the background light line, the gap they bracket, and where
1547 nm sits. Conventions:
  i   : a, b centre-to-centre, r = radius      (period a+b; holes overlap since 2r > b)
  ii  : a, b edge gaps,        r = radius      (period a+b+4r)
  iii : a, b edge gaps,        r = diameter    (period a+b+2r, matches the paper's L column)"""
import sys, json
import meep as mp
from meep import mpb
designs = {'weak': (0.32, 0.15), 'moderate': (0.335, 0.1425), 'strong': (0.35, 0.135)}
w, ncore, r0 = 0.465, 1.81, 0.150
nbg = float(sys.argv[1])
for conv in ['i', 'ii', 'iii']:
    rad = r0 / 2 if conv == 'iii' else r0
    for name, (a, b) in designs.items():
        if conv == 'i':
            L, sep = a + b, a
        else:
            L, sep = a + b + 4 * rad, a + 2 * rad
        x1, x2 = -sep / 2, sep / 2
        Ly = 4.0
        geom = [mp.Block(size=mp.Vector3(mp.inf, w / L, mp.inf), material=mp.Medium(index=ncore)),
                mp.Cylinder(radius=rad / L, center=mp.Vector3(x1 / L, 0), material=mp.air),
                mp.Cylinder(radius=rad / L, center=mp.Vector3(x2 / L, 0), material=mp.air)]
        rowp = {}
        for par in ['yodd', 'yeven']:
            ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(1, Ly / L)), geometry=geom,
                                default_material=mp.Medium(index=nbg), resolution=64,
                                k_points=[mp.Vector3(0.5)], num_bands=4)
            getattr(ms, 'run_te_' + par)()
            fu = [f / L for f in ms.all_freqs[0]]
            fll = 0.5 / L / nbg
            rowp[par] = [round(1e3 / f, 1) for f in fu if f < fll]
        print(json.dumps(dict(nbg=nbg, conv=conv, design=name, L=round(L, 4), lightline_nm=round(2 * L * nbg * 1e3, 1),
                              guided_X_edges_nm=rowp)))
