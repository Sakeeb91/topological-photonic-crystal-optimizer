#!/usr/bin/env bash
# Reproduce the simulation stack used for these benchmarks (no sudo needed).
# micro.mamba.pm and github.com are blocked by this container's proxy; conda.anaconda.org works,
# so micromamba itself is fetched as a conda-forge package tarball.
set -euo pipefail
S=${S:-$(pwd)}
mkdir -p "$S/mamba" && cd "$S/mamba"
curl -sS https://conda.anaconda.org/conda-forge/linux-64/micromamba-2.10.0-0.tar.bz2 -o mm.tar.bz2
tar -xjf mm.tar.bz2 bin/micromamba && rm mm.tar.bz2
export MAMBA_ROOT_PREFIX="$S/mamba/root"
CA_OPT=""; [ -f /root/.ccr/ca-bundle.crt ] && CA_OPT="--ssl-verify /root/.ccr/ca-bundle.crt"
# serial MEEP 1.35 + MPB 1.12 (python 3.11), ~25 s
./bin/micromamba create -y -p "$S/mamba/envs/pmp" -c conda-forge $CA_OPT \
    python=3.11 "pymeep=*=nompi*" mpb numpy scipy matplotlib h5py
# MPI (mpich) MEEP for multi-core 3D runs (OpenMP threads give no speedup in the nompi build)
./bin/micromamba create -y -p "$S/mamba/envs/pmpi" -c conda-forge $CA_OPT \
    python=3.11 "pymeep=*=mpi_mpich*" mpb numpy
rm -rf "$S/mamba/root/pkgs"
# legume (GME) + ceviche (2D FDFD) install fine with pip on Python 3.13 as well as 3.11
python3 -m venv "$S/venv313" && "$S/venv313/bin/pip" install -q legume-gme ceviche
"$S/mamba/envs/pmp/bin/pip" install -q legume-gme
# usage:  $S/mamba/envs/pmpi/bin/mpirun -np 4 $S/mamba/envs/pmpi/bin/python script.py ...
