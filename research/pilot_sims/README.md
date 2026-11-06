# Pilot simulations (2026-11-05)

Real MEEP/MPB runs on a 4-core, 15 GB cloud container, used for [`docs/LITERATURE_REVIEW.md`](../../docs/LITERATURE_REVIEW.md) section 4.
These scripts are independent of the optimizer code in `src/`.

## Environment

`setup_env.sh` installs micromamba from the conda-forge tarball, then creates two environments:

- `pmp`: serial pymeep 1.35 and MPB 1.12, Python 3.11.
- `pmpi`: the MPICH build, for multi-core 3D runs.

It also installs legume-gme and ceviche with pip. The whole install takes about 3 minutes and uses about 2.4 GB.
MEEP has no pip wheel; use conda-forge.

```bash
S=$PWD/research/pilot_sims bash research/pilot_sims/setup_env.sh
```

## Scripts (outputs in `out/`)

| Script | What it computes | Example |
|---|---|---|
| `a1_mpb_waveguide_neff.py` | TE0 effective index of plain SiN strips (MPB `find_k`) | |
| `a_mpb_ssh_unitcell.py` | 3D zone-edge SSH gap of one unit cell. Cladding options: `sub`, `oxide_air`, `oxide_fill`, `susp` | `python a_mpb_ssh_unitcell.py 0.35 0.15 0.05 0.5 0.4 40 1 oxide_air` |
| `b_meep2d_ring_benchmark.py` | 2D full-ring time per step plus Harminv (repo geometry) | |
| `c_meep3d_nanobeam.py` | 3D straight SSH nanobeam with wall `none`, `A` or `B`, plus Harminv | `mpirun -np 4 python c_meep3d_nanobeam.py susp 0.35 0.15 0.10 0.5 0.4 15 A 20 2 400 0.86 0.2` |
| `c2_meep3d_full_ring_benchmark.py` | 3D full-ring time per step and memory | |
| `d3_meep3d_tapered_wall.py` | 3D tanh-tapered (Jackiw-Rebbi-like) wall | `mpirun -np 4 python d3_meep3d_tapered_wall.py susp 0.35 0.15 0.10 0.5 0.4 20 T6 20 2 500 0.86 0.2` |
| `e_meep_cyl_bare_ring_Q.py` | Exact 3D bend-loss Q of a bare ring (cylindrical coordinates, fixed m) | `python e_meep_cyl_bare_ring_Q.py 10 1.6 0.8 oxide 75 30 800` |
| `d2_legume_gme_domainwall.py` | GME cross-check. Failed for narrow strips; kept for the record | |
| `f_mpb2d_published_design_conventions.py` | 2D MPB of the Hotte-Kilburn & Bianucci (J. Opt. 2024) designs under three geometry conventions | `python f_mpb2d_published_design_conventions.py 1.0` |

## Caveats

- Thickness t = 0.4 um is assumed.
- Nanobeam Q is total Q; mirror leakage is not separated from radiation.
- The 3D domain-wall geometries with r = 0.07-0.10 and b = 0.15 have webs of 10 nm down to -50 nm, so they cannot be fabricated. Use them for scaling, not as designs.
- Disorder below about 3 nm is sub-pixel at res 20-40.
