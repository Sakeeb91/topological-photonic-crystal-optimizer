# Literature Review, Research Gaps and a Simulation Plan

**Date:** 2026-11-05
**Scope:** SSH ("topological") photonic-crystal ring resonators and the ML optimization of their Q under fabrication disorder. Covers what this repo does, what the literature has done, which gaps are real, what can be simulated on a 4-core / 15 GB machine, and whether those simulations would add anything to the literature.

**How this was produced.**
- Six parallel literature sweeps used web search; 131 papers were collected and 127 of them were confirmed against a landing page.
- A physics audit read the whole repo.
- An install-and-benchmark run used real MEEP 1.35 and MPB 1.12 on this container.
- Six candidate studies were each checked by an adversarial novelty referee and a compute-feasibility referee.
- A completeness critic reviewed the whole package.
- The full bibliography is in [Appendix A](#appendix-a-bibliography). Unconfirmed items are marked there.
- Every simulation script and log is in [`research/pilot_sims/`](../research/pilot_sims/).

---

## 1. Bottom line

1. **The repo's device is not topological.**
   - `src/simulation_wrapper.py:190-246` wraps a single uniform a-b-a-b lattice around the whole ring.
   - An SSH chain with periodic boundary conditions has no domain wall, so it has no edge or interface state.
   - The source thesis says this outright: *"if a single type of unit cell (either trivial or topological) is present in the ring, no edge state will be supported"* (Hotte-Kilburn 2021).
   - What the code builds is an ordinary photonic-crystal ring (PhCR). NIST already measures those at Q = 1e5 to 1e7.
2. **Every published number in this repo is from the mock.** That includes Q = 55,700, "67% improvement" and "optimal r = 0.15 um". The pilot runs below show the geometry cannot get anywhere near those values.
   - The waveguide in the parameter box does not guide on SiO2 with an air top: n_eff = 1.430 < 1.46.
   - With an oxide top cladding it does guide, but bend loss alone caps Q at about **74** (R = 10 um) to **157** (R = 15 um).
3. **The research question underneath is good, and it is partly open.** The question: does SSH topology buy disorder-tolerant Q in a real 3D ring?
   - The only quantitative SSH-ring study (Hotte-Kilburn & Bianucci, *J. Opt.* 2024) is entirely 2D.
   - It compares against a crude, unoptimized trivial control.
   - It never separates disorder types or reports Q statistics in absolute nanometres.
   - The adjacent literature (Rosiek 2023, Arregui 2021, Barczyk 2022, Minkov 2013) predicts the answer is mostly "no" for Q and "partly" for wavelength. Nobody has run the matched test.
4. **None of the six candidate studies came back as fully novel.** All six were rated "partially novel". Value has to come from two things. One is rigorous, matched, 3D, fabrication-constrained quantification. The other is the one gap simulation cannot close: **no one has measured an SSH domain wall inside a single microring.**
5. **Recommended program** (Section 6):
   - Paper A, about 2-4 weeks of compute here: a 3D, fabrication-constrained test of whether SSH topology helps disorder-limited Q, with a matched trivial control.
   - Paper B: a foundry-ready design sent to a multi-project wafer (MPW) run, then measured with a collaborator.
   - The repo's Bayesian optimization (BO) and NSGA code is reused as the matching and risk-aware optimizer, not as the headline.

---

## 2. What the repo actually is (physics audit)

The full audit with file:line references is summarised here; the severity labels are the auditor's.

| # | Problem | Where | Severity |
|---|---|---|---|
| 1 | Uniform dimerized ring, no domain wall, so no topological state; docs claim "edge-state cavity" | `src/simulation_wrapper.py:209-246` vs `docs/OPTIMIZATION_REPORT.md:128`, `docs/RESEARCH_REPORT.md:58,84-87` | fatal to claims |
| 2 | Q is a closed-form mock with optima typed in (Gaussian peak at \|a-b\| = 0.15, r = 0.12; `optimal_r = 0.15`; -100 Q per hole) | `src/simulation_wrapper.py:125-139, 406-407, 441-468` | fatal |
| 3 | The "MEEP" path generates disordered radii and then ignores them; spread comes from `rng.normal` noise added to Q | `src/simulation_wrapper.py:330,344,406-407` | fatal |
| 4 | "Bandgap" = 50 x dimerization, "mode volume" = 0.5 pi r^2 + 0.1: proxies that create a fake trade-off | `src/simulation_wrapper.py:82-83,136,139`; `src/multi_objective_optimizer.py:341-344` | fatal |
| 5 | Commented MEEP plan is 2D, with the source and monitor at 0.9R / 0.95R (inside the air core, not the waveguide), a broadband source spanning about 10 FSRs, and a run length of about 2% of one photon lifetime | `src/simulation_wrapper.py:350-394` | major |
| 6 | Disorder is relative (% of r), hole radius only; there is no position, sidewall or global-bias disorder | `src/simulation_wrapper.py:223-236,330` | major |
| 7 | `mean(Q) - 2 std(Q)` over 10 samples; Q is heavy-tailed and losses add as 1/Q | `src/simulation_wrapper.py:41-52` | major |
| 8 | `w` never enters the multi-objective Q model; the `coupling_*` variables are dead | `src/simulation_wrapper.py:111-139`; `src/multi_objective_optimizer.py:90-98` | major |
| 9 | `ActiveLearningOptimizer` has unpacking/aliasing bugs (passes `N_cells` as `w`), its multi-fidelity setting has no effect, and nothing calls it | `src/active_learning.py:321,501,599-611` | major |
| 10 | Resolution 20-50 px/um cannot resolve 1.5-3.5 nm disorder (0.08-0.18 px) | `configs/*.yaml` | major |

**Reusable for real research:**
- The trigonometric hole placement.
- `ExtendedDesignParameters`, whose integer `N_cells` closure (R = N L / 2 pi) is the right ring parametrization.
- The pymoo constrained problem.
- `utils.hole_clearance_violations`, config validation and seeding.
- The skopt loop with CSV logging.
- The test suite and CI.

---

## 3. Literature review

### 3.1 The source work and its lineage (Concordia, Bianucci group)

- **Hotte-Kilburn, M.Sc. thesis (Concordia 2021)** and **Hotte-Kilburn & Bianucci, *J. Opt.* 26, 065006 (2024)**.
  - Device: an SSH dimerized PhC waveguide closed into a ring with **two** trivial/topological interfaces.
  - Methods: MPB bands; Q from 2D MEEP with an effective index of 1.81 and Harminv; 10 Gaussian hole-radius realizations per disorder level.
  - Designs (weak / moderate / strong):

    | Quantity | Weak | Moderate | Strong |
    |---|---|---|---|
    | a (um) | 0.32 | 0.335 | 0.35 |
    | b (um) | 0.15 | 0.1425 | 0.135 |
    | Gap (nm) | 7 | 17.7 | 28 |
    | 2D Q | 24,840 | 40,502 | 28,820 |
    | Q flat up to radius disorder of | 3.8% | 5.6% | 7.7% |

    All three use r = 0.150 um and w = 0.465 um.
  - Flux analysis shows that **only leakage along the lattice is protected; radiation is not**.
  - This is the baseline any new result must beat. No fabrication, no 3D.
- **Hotte-Kilburn & Bianucci, *Adv. Phys. X* (2025):** a review of 1D integrated topological photonics.
- **Hotte-Kilburn, Ph.D. thesis (Concordia, F2026):**
  - Fits tight-binding/SSH models to MPB bands.
  - Shows that eigenvalues alone do not fix the SSH invariant; eigenvector or phase data are needed.
  - Per the abstract, it has no new SSH-ring Q.
  - **The full text could not be read here.** It is the main novelty risk for Paper A. Get it.
- **McGarvey-Lechable, Ph.D. thesis (Concordia 2021), Ch. 5:**
  - An earlier SSH-ring design (R = 9.90 um, r = 78 nm, cell 622 nm, 2D).
  - Shows the two edge states splitting as the trivial segment shrinks.
  - Proposes the pair as a "coupled without the need for any external tuning" element.
- **McGarvey-Lechable et al., *Opt. Express* 25, 3916 (2017):** measured uniform hole PhC rings (SOI, 193 nm DUV, IMEC) with intrinsic Q of about 8.4e4. The same group has foundry access.

**Geometry convention problem.**
- In the published table, L = a + b + 2r for all three designs, and r = 150 nm holes would overlap at b = 135 nm centre-to-centre. So a and b must be edge gaps, not centre spacings.
- A quick 2D MPB check run for this review ([`f_mpb2d_published_design_conventions.py`](../research/pilot_sims/f_mpb2d_published_design_conventions.py)) used n = 1.81, w = 0.465 and TE polarization. It could not put a first-order gap at 1547 nm under any of three readings:
  - (iii) edge gaps, r = diameter: the gap sits at 1816-1878 nm and is 32-38 nm wide.
  - (i) centre spacings: the gap sits at 995-1168 nm, and the holes overlap.
- The published table is therefore underspecified (background index, polarization or band order). Reproducing it needs the authors' settings; their paper says "data available upon reasonable request".

### 3.2 Photonic-crystal microrings: what the repo's uniform ring actually is

The uniform dimerized ring is a PhCR: a Bragg-matched modulation that splits a chosen CW/CCW mode pair and creates slow light. This is a mature, high-performance field.

- **Lu, McClung, Srinivasan, *Nat. Photon.* 16, 66 (2022):**
  - SiN "microgear" with inner-sidewall modulation.
  - Slow-light Q = 1.1e6; defect-localized Q = 5.6e5, with more than 10x smaller mode volume.
- Follow-ups:
  - band flipping and gap closing (*Opt. Express* 32, 20360, 2024);
  - highly twisted light (*Nat. Commun.* 14, 1119, 2023);
  - rod/slit rings for cavity QED (*Nanophotonics* 12, 521, 2023);
  - grating-induced loss spectra (Pimbi et al., *Photon. Res.* 2025).
- **Yu, ..., Papp, *Nat. Photon.* 15, 461 (2021):** "edgeless" PhC resonators for spontaneous pulse formation. Black et al., *Optica* 9, 1183 (2022), show optical parametric oscillation (OPO) in PhCRs.
- **ML-designed PhCR:** Liu, Long, Fu et al., *JLT* (2024). Silicon PhC ring, R = 2.1 um, measured Q = 7.67e5. This already takes "first ML-optimized PhC ring".

**Implication.** A uniform-lattice paper from this repo competes with NIST-scale experiments and has nothing topological to offer.

### 3.3 1D topological cavities (SSH / Zak phase)

| Work | Platform | Q (sim / measured) | Note |
|---|---|---|---|
| Blanco-Redondo et al., *PRL* 116, 163901 (2016) | Si waveguide SSH | n/a | Topological vs trivial defect transition |
| Ota et al., *Commun. Phys.* 1, 86 (2018) | GaAs airbridge nanobeam, 3D | 5.97e4 / >= 2e3 | Interior Q optimum vs dimerization |
| Zhang, Xie et al., *LSA* 9, 109 (2020) | GaAs corner state | 5e4 / 2.5-5e3 | 2D corner cavity |
| Gong et al., *Sci. Rep.* 11, 1055 (2021) | SSH array of nanobeam cavities, 3D FDTD | 7.1e4 | Coupled-cavity SSH |
| Scherrer et al., *ACS Photonics* (2024) | InP, telecom | 3D | Topological interface laser |
| arXiv:2608.16138 (2026) | Si PhC slab, valley x SSH interfaces | 5.7e4 / 1.2e4 | Larger gap gives stronger localization and lower Q_rad. **Authorship conflicts between sources** |
| Sanchez-Sanchez, Cheben, Alonso-Ramos et al., arXiv:2609.18362 (2026) | Si Bragg gratings | measured | Band-inversion (no period shift) topological cavities |
| Quan & Loncar, *Opt. Express* 19, 18529 (2011) | Trivial deterministic nanobeam | >1e9 (sim) | The bar any "topological Q" claim faces |

**Pattern.**
- Topological 1D/0D cavities sit at Q of about 1e4-6e4 in simulation and 2e3-1.2e4 in measurement.
- Optimized trivial cavities are 2-5 orders of magnitude higher.
- Topology has never been the route to record Q.

### 3.4 What topological protection protects

- **Rosiek et al., *Nat. Photon.* (2023):** measured strong backscattering in valley-Hall interface modes. There is no protection against fabrication-disorder loss in all-dielectric Si PhCs.
- **Arregui et al., *PRL* 126, 027403 (2021):** at equal group index, topological slow light is not more disorder-tolerant than conventional slow light.
- **Barczyk et al., *LPR* 16, 2200071 (2022):** radiation leakage and protection in topological PhC cavities are separate channels.
- **Khanikaev & Alu, *Nat. Commun.* 15, 931 (2024)** and **Leykam, Xue, Zhang, Chong, *Nat. Rev. Phys.* (2025):** an invariant that survives disorder does not make every mode property survive it.
- **Hu, ..., Song, arXiv:2610.02758 (2026):** there is no universal backscattering immunity in reciprocal waveguides. Protection exists only relative to a stated disorder class.
- **Liu, Zhan, Chen, Li, Xiao, *PRA* 114, 043537 (2026)**, on a full-wave Si SSH waveguide array:
  - gap (hopping) disorder leaves the edge index pinned;
  - on-site disorder of equal strength moves it about 30x more;
  - next-nearest-neighbour (NNN) coupling breaks chiral symmetry.
  - (Flagged by the critic for re-verification.)
- **SSH theory with long-range hopping** (Perez-Gonzalez et al., *PRB* 99, 035146, 2019): even hoppings break chiral symmetry, so diagonal and off-diagonal disorder act alike.

**Consequences for this project.**
- Hole-radius disorder is mostly *on-site* disorder, which breaks chiral symmetry. It is the wrong disorder class to show protection with.
- In the 3D pilot runs, the A- and B-type wall states sit **40-55% of the half-gap away from mid-gap**. So PhC SSH walls are not chiral zero modes to begin with.

### 3.5 Disorder statistics: the method a reviewer will expect

- **Measured disorder in Si PhCs is about 0.4-1.2 nm.**
  - Hagino et al., *PRB* 79, 085112 (2009): radius and position sigma about 1 nm.
  - Le Thomas et al. (2011): radius 0.4 nm and position <0.6 nm in W1 waveguides.
  - Portalupi et al., *PRB* 84, 045423 (2011): residual disorder 0.6-1.1 nm.
- **Large-N statistics:**
  - Taguchi et al., *Opt. Express* (2011): N = 80 devices, mean Q = 3.0e6, sigma_lambda = 0.33 nm.
  - Ashida et al.: 300 mm ArF-immersion lithography, mean Q = 1.5e6.
- **Minkov, Dharanipathy, Houdre, Savona, *Opt. Express* 21, 28233 (2013).**
  - Q^-1 = Q_design^-1 + Q_dis^-1, with <1/Q_dis> proportional to sigma^2.
  - In the disorder-dominated regime, the loss is roughly design-independent across six cavities whose nominal Q spans 5.4e4-7.5e7.
  - This is the null hypothesis any "topology helps Q" claim must beat.
- **SiN sidewall roughness:** Roberts et al., *Adv. Opt. Mater.* 10, 2102073 (2022), give sigma_RMS = 0.53-2.83 nm and correlation length Lc = 47-146 nm. That is comparable to the hole pitch, so correlated disorder matters.
- **Backscatter in rings:** Morichetti et al., *PRL* 104, 033902 (2010); Li et al., *LPR* 10, 420 (2016). CW/CCW splitting scales with n_g^2, which matters at a PhC band edge.
- **Correlated, wafer-scale variation:** Z. Lu et al., *Opt. Express* 25, 9712 (2017).

**What follows from this.**
- Disorder must be specified in absolute nm and include position and correlated components.
- Hotte-Kilburn's 7.7% threshold is about 11.5 nm rms on a 150 nm radius, which is 4-10x above realistic lithography error. **Realistic disorder sits inside their "protected" plateau.**
- The interesting regime is sigma = 1-3 nm.

### 3.6 ML, inverse design and optimization under uncertainty

- **PhC cavity optimization with real solvers:**
  - Minkov & Savona, *Sci. Rep.* 4, 5124 (2014): genetic algorithm + guided-mode expansion (GME), L3 Q = 4.2e6 (FDTD).
  - Vasco & Savona, *Sci. Rep.* 11, 10121 (2021): Si/SiO2 L3 at Q = 4.33e7 after about 8e5 GME evaluations.
  - Asano & Noda, *Opt. Express* 26, 32704 (2018): deep learning, Q = 1.58e9.
  - Minkov et al., *ACS Photonics* 7, 1729 (2020): legume, automatic-differentiation GME.
- **Topological inverse design:**
  - Pilozzi et al., *Commun. Phys.* (2018): ML inverse problem for topological photonics.
  - Long et al., *APL* (2019): Zak-phase targets.
  - Christiansen, Wang, Sigmund, *PRL* (2019): topological insulators by topology optimization.
  - Christiansen et al. (2019): photonic topological insulators with quantitative directionality.
- **Fabrication-aware optimization:**
  - Wang, Lazarov, Sigmund (2011): eroded/dilated worst-case topology optimization.
  - Elbek et al.: stochastic design under fabrication uncertainty.
  - **Plock et al., *JOSA B* (2024):** a nanobeam optimized for nominal Purcell factor falls from 256.5 to 0.7 under 16.8 nm uncertainty. Their uncertainty-aware design holds 3.6.
  - Abulnaga et al., *Nanophotonics* (2025): FDTD Monte Carlo per error class for trivial nanobeams.
- **Risk-aware BO:**
  - Cakmak et al., NeurIPS 2020 (VaR/CVaR, BoRisk);
  - Bogunovic et al., NeurIPS 2018 (StableOpt);
  - Daulton et al., ICML 2022 (MVaR);
  - Oliveira et al., AISTATS 2019.
- **Multi-fidelity BO:** Kim et al., *Digital Discovery* 3, 381 (2024) (Multi-BOWS); Song et al., arXiv:1811.07707.
- **Adiabatic topological interfaces:** Vakulenko et al., *Nat. Commun.* 14, 4629 (2023), get about 19x higher Q from adiabatic interfaces.

**What follows from this.**
- "BO of mean - k std of Q" is an outdated robustness formulation. Quantile, CVaR or harmonic-mean Q with common random numbers is the current method.
- It has not been applied to an SSH interface cavity with a real solver. That is a gap, but a narrow one: the Plock 2024 and Vakulenko 2023 results transfer almost directly.

### 3.7 Platform and fabrication access

From the completeness critic, checked on the vendor pages:

| Platform | Stack | Min feature / gap | Notes |
|---|---|---|---|
| Applied Nanotools (ANT) SiN MPW | 400 nm SiN on oxide, 100 keV EBL | 120 nm / 120 nm | Oxide cladding fills holes; runs about every 2 months; next deadline 2026-11-10; CMC discount for Canadian academics |
| ANT Si (NanoSOI) | 220 nm SOI | 60 nm | Bends of R = 5-10 um are nearly lossless |

**Consequences.**
- At a first-order Bragg period of about 0.5 um, the SiN 120-nm rule forces a and b to at least 0.24 um each, which removes almost all dimerization.
- An SSH hole lattice at 1.55 um is therefore **barely realizable on the open SiN MPW**. Si (60 nm) is the realistic experimental route.
- Every 3D pilot Q below was computed on geometry with webs of 10 nm down to -50 nm (overlapping holes). The pilot Q values show scaling but do not transfer to a fabricable device.

---

## 4. Pilot simulations run on this machine

Setup: MEEP 1.35 and MPB 1.12 via micromamba from conda-forge (serial and MPICH builds), plus legume and ceviche via pip. The thickness t = 0.4 um is an assumption, since the repo has none. Scripts and logs are in `research/pilot_sims/`; `setup_env.sh` rebuilds the stack in about 3 minutes.

| Test | Result | Takeaway |
|---|---|---|
| TE0 n_eff, SiN 0.5 x 0.4 um, SiO2 below / air top (MPB) | 1.430 (<1.46) | **Repo waveguide is not guided** in that stack; needs an oxide top cladding (1.523) or w >= 0.7 |
| Bare-ring bend-loss Q, exact 3D cylindrical MEEP, oxide-clad | R=10, w=0.5: **74**; R=15, w=0.5: 157; R=10, w=1.0: 1.6e3; R=15, w=1.0: 3.8e4; R=10, w=1.0, t=0.8: 3.2e5; R=10, w=1.6, t=0.8: 2.0e6 | The repo's parameter box is capped at Q of about 1e2 **before any holes are added** |
| SSH unit-cell gap, oxide-clad air holes, w=0.5, a/b=0.35/0.15 (3D MPB) | r=0.05: 1492-1517 nm (25.5 nm, 1.69%); r=0.07: 36.8 nm; a=b: gap closes (0.07 nm) | A genuine SSH gap, but 30-55 nm blue of 1547 nm |
| 2D MEEP full ring, repo geometry | Q = 270-1300 with holes vs 1.4e4-6e4 without | Even the repo's own planned 2D model gives Q 40-200x below the mock |
| 3D straight nanobeam, single abrupt wall (suspended, r=0.10) | Wall A: Q = 1280 (res 30: 1297); Wall B: Q = 73 | A and B walls are **inequivalent** (17x in Q, about 10% in frequency) |
| 3D tanh-tapered wall, same peak dimerization | M=3: Q = 748; M=6: Q = 2667 | Taper gives 10-36x; consistent with Vakulenko 2023 |
| 3D oxide-clad wall A, r=0.07 | N=30: Q = 228; N=50: Q = 853 | Localization length about 30 periods (about 15 um), roughly 1/4 of an R=10 circumference, so the two ring walls will hybridize |
| legume GME supercell of the domain wall | failed (spurious supercell modes) | GME is a poor fit for narrow strips; use 3D MEEP |
| 2D MPB of published designs, three conventions | No first-order gap at 1547 nm under any reading tried | Published geometry is underspecified; needs author settings |

Caveats:
- The nanobeam Q values are total Q; flux monitors did not separate radiation from mirror leakage.
- Holes with r = 0.05 are 1-2 pixels at res 20-40.
- 2.5 nm disorder is about 1/20 of a pixel. Disorder at 1-3 nm needs perturbation theory or res >= 60.
- The bend-loss Q comes from one solver and should be cross-checked with an independent bend-mode solver before it headlines anything.

---

## 5. What you can run yourself: cost table

Measured or extrapolated on this container (4 cores, 15 GB). Use the MPI build (`mpirun -np 4`) for 3D. The serial build gets no OpenMP speedup.

| Simulation | Wall time here | Feasible |
|---|---|---|
| MPB unit-cell gap (3D strip supercell, 1 k-point) | 5-20 s; full band diagram 1-3 min | yes |
| Cylindrical bend-loss Q of a bare ring, any (R, w, t, cladding) | 1-2 min; run 4 at once | yes |
| 3D nanobeam with a domain wall, Q up to about 1e4, res 20 | 3-15 min (about 20 min at res 30) | yes |
| Disorder ensemble, 50-100 realizations of one 3D nanobeam | 3-8 h | yes |
| 2D full ring, Q about 1e4 (res 20 / res 40) | 2.5 / 17 min | yes, but physically inadequate |
| Full 3D ring R=10, res 20, Q about 1e4 | 4-5 h; 1.7 GB | yes, a handful of validation runs |
| Full 3D ring, Q 1e5-1e6, or res >= 30, or any ensemble / BO loop | 2 days to weeks per run | no: needs HPC or GPU FDTD (Tidy3D) |

The critic noted two refinements:
- Harminv's filter diagonalization recovers Q well above pi f T for an isolated mode, so the 3D-ring estimates may be 3-10x pessimistic.
- The exception is resolving CW/CCW or A/B doublets: budget T > 1/delta f.

**Workflow that is both physically sound and feasible here:**
1. MPB gives gaps, band edges and Zak-phase parity (seconds per design).
2. Cylindrical MEEP picks a cross-section with Q_bend > 1e5.
3. 3D MEEP straight nanobeams and unrolled-ring supercells give wall frequency, Q_rad and disorder ensembles (minutes per run).
4. Combine them in a Q budget: 1/Q = 1/Q_bend + 1/Q_rad,wall + 1/Q_scatter + 1/Q_abs, with the wall-wall splitting reported separately.
5. Validate with 1-2 full 3D rings.

---

## 6. Do these simulations contribute? Candidate studies and verdicts

Each candidate went through an adversarial novelty search and a compute and physics check.

| ID | Study | Novelty verdict | Value | Feasible here | Main problem |
|---|---|---|---|---|---|
| C1 | 3D Q budget and realizability of the published SSH-ring designs (corrected reproduction) | partially novel | medium | yes, about 3-5 days | Conceptual result ("radiation unprotected") already in the source paper; a ring always has one A and one B wall, so wall type is not a free choice; "Q_lattice" is not a loss channel in a closed ring |
| C2 | Chiral-preserving vs chiral-breaking disorder classes in 3D SiN SSH nanobeams | partially novel | medium (critic: low-medium) | yes, about 3-4 days | Transplant of Liu *PRA* 2026; PhC walls are not mid-gap, so the "chiral" label does not carry over; expect a ratio of about 1-3, not >= 5 |
| C3 | Matched, statistically powered comparison: SSH walls vs an optimized trivial (Quan-Loncar) cavity | partially novel | medium | reduced scope; 190-230 h as written | H0 is essentially Minkov 2013; all pilot geometries break the 50-nm web rule, so matching must be redone |
| C4 | Closing the ring: two-wall hybridization, A/B inequivalence, Q budget | partially novel | medium (critic: low-medium) | yes, 2-3 days compute | Two-wall splitting and "tuning-free pair" are in McGarvey 2021; A and B walls cannot hybridize (26 nm detuned) |
| C5 | Risk-aware multi-fidelity BO of the domain-wall profile (Pareto: Q_rad vs disorder loss vs gap) | partially novel | medium (critic: low-medium) | reduced scope; 5-6 weeks as written | Trade-off is in Hotte-Kilburn 2024, Ota 2018, arXiv:2608.16138; taper gain in Vakulenko 2023; collapse of the nominal optimum in Plock 2024; ML PhC ring in Liu *JLT* 2024 |
| C6 | Band-inversion (Jackiw-Rebbi) wall in a SiN microgear ring | partially novel | low | reduced scope | Scooped in straight guides by Sanchez-Sanchez 2026 and Randerson *LSA* 2026; physically a lambda/4-shifted DFB (Haus-Shank 1976) |
| C0 | Fabrication-constrained, foundry-ready SSH-ring design + MPW tapeout + measurement | **only "high-confidence" gap** | **high** | design: about 1 week compute; fab: external | Needs a fab budget and a test collaborator |

**Straight answer to "would my simulations contribute?"**
- **Repo as it stands:** no. None of its numbers are physical.
- **Running MEEP on the repo's geometry and re-running the BO:** no. The geometry is non-topological and bend-limited, and uniform PhC rings are an established field.
- **Simulation-only work with the corrections above:** yes, as a solid mid-tier paper (*J. Opt.*, *Optics Express*, *JOSA B*, *Optics Letters*, *PRA*). The contribution is the first 3D, fabrication-constrained, statistically powered answer to "does SSH topology help disorder-limited Q in a ring?", with a mechanism. On current evidence the expected answer is a well-characterized null for Q, plus a partial win for wavelength under some disorder classes. A clean null with a mechanism is publishable; a vague null is not.
- **Design plus tapeout plus measurement:** yes, and a strong one. It would be the first measured SSH domain-wall state in a single microring.

---

## 7. Recommended program

### Step 0: Fix the premise (1-2 days, not publishable, required)
1. Replace `_generate_ssh_ring_geometry` with a domain-wall builder.
   - Use segments of (a,b) and (b,a) cells, an integer `N_cells`, and R = N L / 2 pi.
   - Drop the a/b rescaling.
   - Note that a two-segment ring always contains one A-type and one B-type wall.
2. Make disorder absolute (nm), with separate classes: radius, longitudinal position, transverse offset, global CD bias, and correlated sidewall roughness.
3. Replace `mean - 2 std` with Q10 (10th-percentile Q), the harmonic mean 1/<1/Q>, or P(Q > Q_spec), and use common random numbers across designs.
4. Wire `evaluate_design_meep` to the pilot scripts (MPB, cylindrical, 3D nanobeam) as fidelity levels.
5. Delete or clearly label every mock-derived number in `README.md` and `docs/`.

### Paper A: "Does SSH topology buy disorder-tolerant Q? A 3D, fabrication-constrained test in photonic-crystal cavities and rings" (about 2-4 weeks of this machine)

The core is C3 + C1, with C4 and C2 as sections.

**Inputs to get first:**
- the full text of the 2026 Ph.D. thesis;
- the exact 2D settings for J. Opt. 2024 (email the authors);
- a target platform: ANT Si 220 nm (60 nm rule) as primary, with SiN 400 nm as the stretch platform.

**Studies:**
1. **3D Q budget of the published designs** on real stacks. Headline: the 2D-to-3D Q ratio, plus thresholds re-expressed in nm and compared with 1-3 nm lithography error.
2. **Matched families** at equal nominal Q, V and lambda, all inside the fabrication rules. Use the repo's BO or a 1D sweep to do the matching.
   - (T1) abrupt SSH wall;
   - (T2) tanh-tapered SSH wall, certified topological by MPB band-edge parity;
   - (Tr) Quan-Loncar Gaussian-mirror trivial cavity;
   - (Tr0) the paper's one-hole control.
3. **Disorder ensembles** (N >= 48 per cell, common random numbers):
   - First-order sensitivities d omega/dr_i and d omega/dx_i from the shifting-boundary perturbation formula (Johnson et al., *PRE* 65, 066611, 2002) give frequency statistics at 0.5-3 nm in seconds.
   - FDTD validation at 5-10 nm.
   - Report C in <1/Q_d> = C sigma^2, Q10, the harmonic mean and sigma_lambda, per class, with bootstrap confidence intervals.
4. **Mechanism:** the Minkov split. Does the SSH wall change only Q_design (washed out by disorder), or also the waveguide-mode scattering term?
5. **Ring closure:** A/B wall detuning and Q asymmetry in an unrolled-ring 3D supercell, the hybridization scaling, the minimum R for isolated walls, and 1-2 full 3D ring validations.

**Pre-registered predictions:**
- Q: no advantage for T1/T2 over Tr at matched V.
- sigma_lambda: smaller only under position disorder, by about 1-3x.
- Tr0 loses, which reproduces the published claim; that win disappears against Tr.

### Paper B / experiment (C0): highest value

- Design an AA/BB-symmetric wall-pair ring, a matched trivial control, and deliberate disorder splits (radius vs position at 0 / 2 / 5 nm) across several R values.
- Target ANT Si 220 nm rules.
- Validate it with the Paper A pipeline.
- Submit to an MPW and measure with a test lab. The Bianucci group at Concordia is the natural partner.
- The 2026-11-10 SiN deadline is too close; plan for the following run.
- Cost: unknown until ANT quotes.

### Deprioritize
- A standalone "risk-aware BO" paper (C5): the optimizer is not the novelty.
- C6: scooped.
- Uniform-PhC-ring work: NIST owns it.
- If cloud credits are available, a GPU FDTD service such as Tidy3D makes full 3D ring ensembles practical. That would turn study 5 of Paper A into a full statistical study.

---

## 8. Open verification items (fix before citing)

- **arXiv:2608.16138 authorship:** sources disagree (Uemura, Dai, Moritake, Ono, Kuramochi, Notomi vs Akahane, Asano, Song, Noda). arxiv.org did not resolve from this container.
- **Liu et al., *PRA* 114, 043537 (2026)** and **Hu et al., arXiv:2610.02758:** marked verified by one agent and not by another. C2's verdict depends on Liu.
- **Abulnaga et al., *Nanophotonics* (2025):** check DOI 10.1515/nanoph-2024-0500.
- **Unconfirmed:**
  - Gao et al., *Sci. Rep.* 6, 19999 (2016);
  - Wang et al., *PRL* 129, 186101 (2022) (author order differs between sources);
  - Liu et al., *Opt. Lett.* 50, 2570 (2025);
  - Gong vs Barczyk authorship of arXiv:2202.07620.
- **Hotte-Kilburn Ph.D. thesis (2026):** only the abstract was read.
- **ANT SiN BOX thickness:** 4.5 um on the spec page vs 3 um on the order page.
- **Not verified in this pass, but a referee will expect them distinguished:**
  - 2D topological ring lattices: Hafezi et al. 2013; Mittal et al. 2018; Bandres et al. and Harari et al. 2018.
  - Topological frequency combs: Mittal et al., *Nat. Phys.* 2021; Flower et al., *Science* 2024.
  - Valley-Hall ring resonators: Barik et al. 2020; Jalali Mehrabad et al. 2020; Zeng et al. 2020.
  - Synthetic-dimension rings: Dutt, Yuan, Fan.
  - Johnson et al., *Appl. Phys. B* 81, 283 (2005), on roughness losses.

---

## Appendix A: Bibliography

The literature agents collected these works. Each was confirmed against a DOI, journal or arXiv landing page found by search, unless it is marked otherwise. Grouping follows the search slice that found it, so some cross-cutting papers appear in an earlier group.

### A. Direct prior art: SSH and photonic-crystal rings

- K. McGarvey-Lechable and P. Bianucci, 'Maximizing slow-light enhancement in one-dimensional photonic crystal ring resonators', Opt. Express 22, 26032-26041 (2014). <https://doi.org/10.1364/OE.22.026032>
- K. McGarvey-Lechable, T. Hamidfar, D. Patel, L. Xu, D. V. Plant, P. Bianucci, 'Slow light in mass-produced, dispersion-engineered photonic crystal ring resonators', Opt. Express 25, 3916-3926 (2017). <https://doi.org/10.1364/OE.25.003916>
- K. McGarvey-Lechable and P. Bianucci, 'Bloch-Floquet waves in optical ring resonators', Phys. Rev. B 97, 214204 (2018). <https://doi.org/10.1103/PhysRevB.97.214204>
- Y. Ota, R. Katsumi, K. Watanabe, S. Iwamoto, Y. Arakawa, 'Topological photonic crystal nanocavity laser', Commun. Phys. 1, 86 (2018). <https://www.nature.com/articles/s42005-018-0083-7>
- M. Parto, S. Wittek, H. Hodaei, G. Harari, M. A. Bandres, J. Ren, M. C. Rechtsman, M. Segev, D. N. Christodoulides, M. Khajavikhan, 'Edge-mode lasing in 1D topological active arrays', Phys. Rev. Lett. 120, 113901 (2018). <https://doi.org/10.1103/PhysRevLett.120.113901>
- H. Zhao, P. Miao, M. H. Teimourpour, S. Malzard, R. El-Ganainy, H. Schomerus, L. Feng, 'Topological hybrid silicon microlasers', Nat. Commun. 9, 981 (2018). <https://doi.org/10.1038/s41467-018-03434-2>
- D. Leykam and L. Yuan, 'Topological phases in ring resonators: recent progress and future prospects', Nanophotonics 9, 4473 (2020). <https://doi.org/10.1515/nanoph-2020-0415>
- A. Hotte-Kilburn, 'Topological photonic crystal ring resonators', M.Sc. thesis, Department of Physics, Concordia University, Montreal (Jan. 2021; deposited S2021). Supervisor: Dr. Pablo Bianucci. Committee: S. Maiti (chair), A. Champagne.. <https://spectrum.library.concordia.ca/id/eprint/987925/1/Hotte-Kilburn_Msc_S2021.pdf>
- K. McGarvey-Lechable, 'Manipulating states of light in optical ring resonators', Ph.D. thesis, Concordia University (S2021), supervisor P. Bianucci; Ch. 5 'Topological edge states in photonic crystal ring resonators'. <https://spectrum.library.concordia.ca/id/eprint/988075/1/McGarvey-Lechable_PhD_S2021.pdf>
- S.-P. Yu, D. C. Cole, H. Jung, G. T. Moille, K. Srinivasan, S. B. Papp, 'Spontaneous pulse formation in edgeless photonic crystal resonators', Nat. Photon. 15, 461-467 (2021). <https://doi.org/10.1038/s41566-021-00800-3>
- G. Arregui, J. Gomis-Bresco, C. M. Sotomayor-Torres, P. D. Garcia, 'Quantifying the robustness of topological slow light', Phys. Rev. Lett. 126, 027403 (2021). <https://doi.org/10.1103/PhysRevLett.126.027403>
- X. Lu, A. McClung, K. Srinivasan, 'High-Q slow light and its localization in a photonic crystal microring', Nat. Photon. 16, 66-71 (2022); arXiv:2109.08271. <https://doi.org/10.1038/s41566-021-00912-w>
- J. A. Black, G. M. Brodnik, H. Liu, S.-P. Yu, D. R. Carlson, J. Zang, T. C. Briles, S. B. Papp, 'Optical-parametric oscillation in photonic-crystal ring resonators', Optica 9, 1183-1189 (2022). <https://doi.org/10.1364/OPTICA.469210>
- R. Barczyk, N. Parappurath, S. Arora, T. Bauer, L. Kuipers, E. Verhagen, 'Interplay of leakage radiation and protection in topological photonic crystal cavities', Laser Photonics Rev. 16, 2200071 (2022). <https://doi.org/10.1002/lpor.202200071>
- M. Wang, F. Zhou, X. Lu, A. McClung, M. Davanco, V. A. Aksyuk, K. Srinivasan, 'Fractional optical angular momentum and multi-defect-mediated mode renormalization and orientation control in photonic crystal microring resonators', Phys. Rev. Lett. 129, 186101 (2022). <https://doi.org/10.1103/PhysRevLett.129.186101> *(not confirmed by the search agents)*
- X. Lu, F. Zhou, Y. Sun, A. Chanana, M. Wang, A. McClung, V. A. Aksyuk, M. Davanco, K. Srinivasan, 'Rod and slit photonic crystal microrings for on-chip cavity quantum electrodynamics', Nanophotonics 12, 521-529 (2023). <https://doi.org/10.1515/nanoph-2022-0622>
- X. Lu, M. Wang, F. Zhou, M. Heuck, W. Zhu, V. A. Aksyuk, D. R. Englund, K. Srinivasan, 'Highly-twisted states of light from a high quality factor photonic crystal ring', Nat. Commun. 14, 1119 (2023). <https://doi.org/10.1038/s41467-023-36589-8>
- G. Moille, X. Lu, J. Stone, D. Westly, K. Srinivasan, 'Fourier synthesis dispersion engineering of photonic crystal microrings for broadband frequency combs', Commun. Phys. 6 (2023). <https://doi.org/10.1038/s42005-023-01253-6>
- C. A. Rosiek, G. Arregui, A. Vladimirova, M. Albrechtsen, B. Vosoughi Lahijani, R. E. Christiansen, S. Stobbe, 'Observation of strong backscattering in valley-Hall photonic topological interface modes', Nat. Photon. (2023). <https://doi.org/10.1038/s41566-023-01189-x>
- A. Hotte-Kilburn and P. Bianucci, 'Implementation of the SSH model in an optical ring resonator', J. Opt. 26, 065006 (2024), open access, Focus Issue on Metaphotonics. <https://doi.org/10.1088/2040-8986/ad42af>
- X. Lu, A. Chanana, Y. Sun, A. McClung, M. Davanco, K. Srinivasan, 'Band flipping and bandgap closing in a photonic crystal ring and its applications', Opt. Express 32, 20360 (2024). <https://doi.org/10.1364/OE.521350>
- A. E. Ulanov, T. Wildi, N. G. Pavlov, J. D. Jost, M. Karpov, T. Herr, 'Synthetic reflection self-injection-locked microcombs', Nat. Photon. 18, 294-299 (2024). <https://doi.org/10.1038/s41566-023-01367-x>
- A. Hotte-Kilburn and P. Bianucci, 'Integrated topological photonics in one dimension', Advances in Physics: X (2025), article 2476417. <https://doi.org/10.1080/23746149.2025.2476417>
- D. Pimbi, Y. Sun, R. Zektzer, X. Lu, K. Srinivasan, 'Full spectral response of grating-induced loss in photonic crystal microrings', Photonics Research (2025). <https://doi.org/10.1364/PRJ.568378>
- H. Liu, I. Dickson, A. O. Antohe, L. G. Carpenter, J. Zang, A. R. Carollo, et al., 'Implementing photonic-crystal resonator frequency combs in a photonic foundry', Opt. Lett. 50, 2570 (2025). *(not confirmed by the search agents)*
- A. Hotte-Kilburn, 'Design of Integrated Photonic Realizations of Tight-Binding Hamiltonians', Ph.D. thesis, Concordia University (F2026; PDF dated about 2026-07). <https://spectrum.library.concordia.ca/id/eprint/997934/1/Hotte-Kilburn_PhD_F2026.pdf>
- N. Fried, D. L. P. Vitullo, A. Dutt, 'Multiband topological heterojunctions on the surface nanoscale axial photonics platform', Nat. Commun. (2026). <https://www.nature.com/articles/s41467-026-76041-1>

### B. 1D topological cavities and the robustness debate

- Q. Quan and M. Loncar, "Deterministic design of wavelength scale, ultra-high Q photonic crystal nanobeam cavities", Opt. Express 19, 18529 (2011). <https://doi.org/10.1364/OE.19.018529>
- S. L. Portalupi et al., "Deliberate versus intrinsic disorder in photonic crystal nanocavities investigated by resonant light scattering", Phys. Rev. B 84, 045423 (2011). <https://doi.org/10.1103/PhysRevB.84.045423>
- M. Minkov, U. P. Dharanipathy, R. Houdre, V. Savona, "Statistics of the disorder-induced losses of high-Q photonic crystal cavities", Opt. Express 21, 28233 (2013). <https://doi.org/10.1364/OE.21.028233>
- C. Poli, M. Bellec, U. Kuhl, F. Mortessagne, H. Schomerus, "Selective enhancement of topologically induced interface states in a dielectric resonator chain", Nat. Commun. 6, 6710 (2015). <https://doi.org/10.1038/ncomms7710>
- Y. Yamaguchi, S. Jeon, B.-S. Song, Y. Tanaka, T. Asano, S. Noda, "Analysis of Q-factors of structural imperfections in triangular cross-section nanobeam photonic crystal cavities", J. Opt. Soc. Am. B 32, 1792 (2015). <https://doi.org/10.1364/JOSAB.32.001792>
- A. Blanco-Redondo et al., "Topological Optical Waveguiding in Silicon and the Transition between Topological and Trivial Defect States", Phys. Rev. Lett. 116, 163901 (2016). <https://doi.org/10.1103/PhysRevLett.116.163901>
- P. St-Jean et al., "Lasing in topological edge states of a one-dimensional lattice", Nat. Photon. 11, 651-656 (2017). <https://doi.org/10.1038/s41566-017-0006-2>
- Y. Ota, R. Katsumi, K. Watanabe, S. Iwamoto, Y. Arakawa, "Topological photonic crystal nanocavity laser", Commun. Phys. 1, 86 (2018). <https://doi.org/10.1038/s42005-018-0083-7>
- Y. Ota et al., "Photonic crystal nanocavity based on a topological corner state", Optica 6, 786-789 (2019). <https://doi.org/10.1364/OPTICA.6.000786>
- W. Zhang, X. Xie et al., "Low-threshold topological nanolasers based on the second-order corner state", Light Sci. Appl. 9, 109 (2020). <https://doi.org/10.1038/s41377-020-00352-1>
- X. Xie, W. Zhang et al., "Cavity Quantum Electrodynamics with Second-Order Topological Corner State", Laser Photonics Rev. 14, 1900425 (2020). <https://doi.org/10.1002/lpor.201900425>
- G. Leveque, Y. Pennec, P. Szriftgiser, A. Amo, A. Martinez, "Scattering-matrix approach for a quantitative evaluation of the topological protection in valley photonic crystals", arXiv:2301.10565 (2023). <https://arxiv.org/abs/2301.10565>
- F. Davoodi and N. Talebi, "Topological Plasmonic Ring Resonator", arXiv:2308.11429 (2023). <https://arxiv.org/abs/2308.11429>
- Y. Zhang et al., "High-speed electro-optic modulation in topological interface states of a one-dimensional lattice", Light Sci. Appl. (2023). <https://doi.org/10.1038/s41377-023-01251-x>
- A. B. Khanikaev and A. Alu, "Topological photonics: robustness and beyond", Nat. Commun. 15, 931 (2024). <https://doi.org/10.1038/s41467-024-45194-2>
- F. Davoodi, "From bound states to quantum spin models: chiral coherent dynamics in topological photonic rings", Nanophotonics (2025); arXiv:2507.11271. <https://arxiv.org/abs/2507.11271>
- Z. Liu, X. Zhan, D. Chen, P. Li, J.-J. Xiao, "Sublattice-polarized bound state in the continuum in a silicon waveguide array with approximate chiral symmetry", Phys. Rev. A 114, 043537 (2026). <https://doi.org/10.1103/rsqn-bfgl>
- T. Uemura, W. Dai, Y. Moritake, M. Ono, E. Kuramochi, M. Notomi, "Nanocavity Confinement by Orthogonal Valley- and SSH- Topological Interfaces In Glide-Symmetric Photonic Crystal Structures", arXiv:2608.16138 (2026). <https://arxiv.org/abs/2608.16138>

### C. Fabrication disorder and Q statistics

- D. Gerace and L. C. Andreani, "Effects of disorder on propagation losses and cavity Q-factors in photonic crystal slabs," Photonics Nanostruct. Fundam. Appl. 3, 120-128 (2005). <https://doi.org/10.1016/j.photonics.2005.09.012>
- S. Hughes, L. Ramunno, J. F. Young, J. E. Sipe, "Extrinsic optical scattering loss in photonic crystal waveguides: role of fabrication disorder and photon group velocity," Phys. Rev. Lett. 94, 033903 (2005). <https://doi.org/10.1103/PhysRevLett.94.033903>
- M. Borselli, T. J. Johnson, O. Painter, "Beyond the Rayleigh scattering limit in high-Q silicon microdisks: theory and experiment," Opt. Express 13, 1515-1530 (2005). <https://doi.org/10.1364/OPEX.13.001515>
- T. Asano, B.-S. Song, S. Noda, "Analysis of the experimental Q factors (~1 million) of photonic crystal nanocavities," Opt. Express 14, 1996-2002 (2006). <https://doi.org/10.1364/OE.14.001996>
- H. Hagino, Y. Takahashi, Y. Tanaka, T. Asano, S. Noda, "Effects of fluctuation in air hole radii and positions on optical characteristics in photonic crystal heterostructure nanocavities," Phys. Rev. B 79, 085112 (2009). <https://doi.org/10.1103/PhysRevB.79.085112>
- F. Morichetti, A. Canciamilla, C. Ferrari, M. Torregiani, A. Melloni, M. Martinelli, "Roughness induced backscattering in optical silicon waveguides," Phys. Rev. Lett. 104, 033902 (2010). <https://doi.org/10.1103/PhysRevLett.104.033902>
- F. Morichetti, A. Canciamilla, M. Martinelli, A. Samarelli, R. M. De La Rue, M. Sorel, A. Melloni, "Coherent backscattering in optical microring resonators," Appl. Phys. Lett. 96, 081112 (2010). <https://doi.org/10.1063/1.3330894>
- Y. Taguchi, Y. Takahashi, Y. Sato, T. Asano, S. Noda, "Statistical studies of photonic heterostructure nanocavities with an average Q factor of three million," Opt. Express 19, 11916-11921 (2011). <https://doi.org/10.1364/OE.19.011916>
- N. Le Thomas, Z. Diao, H. Zhang, R. Houdre, "Statistical analysis of subnanometer residual disorder in photonic crystal waveguides: Correlation between slow light properties and structural properties," J. Vac. Sci. Technol. B 29, 051601 (2011). <https://doi.org/10.1116/1.3622289>
- Q. Li, A. A. Eftekhar, M. Sodagar, Z. Xia, A. H. Atabaki, A. Adibi, "Vertical integration of high-Q silicon nitride microresonators into silicon-on-insulator platform," Opt. Express 21, 18236 (2013). <https://doi.org/10.1364/OE.21.018236>
- M. Minkov and V. Savona, "Automated optimization of photonic crystal slab cavities," Sci. Rep. 4, 5124 (2014). <https://www.nature.com/articles/srep05124>
- U. P. Dharanipathy, M. Minkov, M. Tonin, V. Savona, R. Houdre, "High-Q silicon photonic crystal cavity for enhanced optical nonlinearities," Appl. Phys. Lett. (2014); arXiv:1311.0997. <https://arxiv.org/abs/1311.0997>
- A. Li, T. Van Vaerenbergh, P. De Heyn, P. Bienstman, W. Bogaerts, "Backscattering in silicon microring resonators: a quantitative analysis," Laser Photonics Rev. 10, 420-431 (2016). <https://doi.org/10.1002/lpor.201500207>
- T. Asano, Y. Ochi, Y. Takahashi, K. Kishimoto, S. Noda, "Photonic crystal nanocavity with a Q factor exceeding eleven million," Opt. Express 25, 1769-1777 (2017). <https://doi.org/10.1364/OE.25.001769>
- K. Ashida, M. Okano, M. Ohtsuka, M. Seki, N. Yokoyama, K. Koshino, M. Mori, T. Asano, S. Noda, Y. Takahashi, "Ultrahigh-Q photonic crystal nanocavities fabricated by CMOS process technologies," Opt. Express 25, 18165-18174 (2017). <https://doi.org/10.1364/OE.25.018165>
- Z. Lu, J. Jhoja, J. Klein, X. Wang, A. Liu, J. Flueckiger, J. Pond, L. Chrostowski, "Performance prediction for silicon photonics integrated circuits with layout-dependent correlated manufacturing variability," Opt. Express 25, 9712-9733 (2017). <https://doi.org/10.1364/OE.25.009712>
- C. Cui, K. Liu, Z. Zhang, "Chance-constrained and yield-aware optimization of photonic ICs with non-Gaussian correlated process variations," arXiv:1908.07574 (2019); see also PoBO, arXiv:2107.12593. <https://arxiv.org/abs/1908.07574>
- M. Minkov, I. A. D. Williamson, L. C. Andreani, D. Gerace, B. Lou, A. Y. Song, T. W. Hughes, S. Fan, "Inverse design of photonic crystals through automatic differentiation," ACS Photonics 7, 1729-1741 (2020) (legume GME code). <https://doi.org/10.1021/acsphotonics.0c00327>
- S. Roberts, X. Ji, J. Cardenas, M. Corato-Zanarella, M. Lipson, "Measurements and modeling of atomic-scale sidewall roughness and losses in integrated photonic devices," Adv. Opt. Mater. 10, 2102073 (2022); arXiv:2105.11477. <https://doi.org/10.1002/adom.202102073>
- S. Arora, T. Bauer, R. Barczyk, E. Verhagen, L. Kuipers, "Multiple backscattering in trivial and non-trivial topological photonic crystal edge states with controlled disorder," arXiv:2310.02978 (2023). <https://arxiv.org/abs/2310.02978>
- N. Granchi, F. Intonti, M. Florescu, P. D. Garcia, M. Gurioli, G. Arregui, "Q-factor optimization of modes in ordered and disordered photonic systems using non-Hermitian perturbation theory," ACS Photonics 10, 2808-2815 (2023). <https://doi.org/10.1021/acsphotonics.3c00510>
- M. Plock, F. Binkowski, L. Zschiedrich, P.-I. Schneider, S. Burger, "Fabrication uncertainty guided design optimization of a photonic crystal cavity by using Gaussian processes," J. Opt. Soc. Am. B (2024); arXiv:2402.09420. <https://arxiv.org/abs/2402.09420>
- N. K. Vij, P. Purkayastha, J. R. Basani, E. Waks, "Inverse-designed photonic crystal cavities with controllable far-field numerical aperture," arXiv:2509.16827 (2025). <https://arxiv.org/abs/2509.16827>

### D. ML, inverse design and robust optimization

- Wang, F., Lazarov, B. S. & Sigmund, O., "On projection methods, convergence and robust formulations in topology optimization", Struct. Multidisc. Optim. 43, 767-784 (2011). <https://doi.org/10.1007/s00158-010-0602-y>
- Minkov, M. & Savona, V., "Automated optimization of photonic crystal slab cavities", Sci. Rep. 4, 5124 (2014). <https://doi.org/10.1038/srep05124>
- Asano, T. & Noda, S., "Optimization of photonic crystal nanocavities based on deep learning", Opt. Express 26, 32704-32717 (2018). <https://doi.org/10.1364/OE.26.032704>
- Bogunovic, I., Scarlett, J., Jegelka, S. & Cevher, V., "Adversarially Robust Optimization with Gaussian Processes" (StableOpt), NeurIPS 31 (2018). <https://neurips.cc/virtual/2018/poster/11561>
- Song, J., Tokpanov, Y. S., Chen, Y., Fleischman, D., Fountaine, K. T., Atwater, H. A. & Yue, Y., "Optimizing Photonic Nanostructures via Multi-fidelity Gaussian Processes", arXiv:1811.07707 (2018). <https://arxiv.org/abs/1811.07707>
- Pilozzi, L., Farrelly, F. A., Marcucci, G. & Conti, C., "Machine learning inverse problem for topological photonics", Commun. Phys. 1, 57 (2018). <https://doi.org/10.1038/s42005-018-0058-8>
- Oliveira, R., Ott, L. & Ramos, F., "Bayesian optimisation under uncertain inputs", AISTATS 2019, PMLR 89:1177-1184. <https://proceedings.mlr.press/v89/oliveira19a.html>
- Christiansen, R. E., Wang, F., Sigmund, O. & Stobbe, S., "Designing photonic topological insulators with quantum-spin-Hall edge states using topology optimization", Nanophotonics 8, 1363-1369 (2019). <https://doi.org/10.1515/nanoph-2019-0057>
- Christiansen, R. E., Wang, F. & Sigmund, O., "Topological Insulators by Topology Optimization", Phys. Rev. Lett. 122, 234502 (2019). <https://doi.org/10.1103/PhysRevLett.122.234502> *(not confirmed by the search agents)*
- Long, Y., Ren, J., Li, Y. & Chen, H., "Inverse design of photonic topological state via machine learning", Appl. Phys. Lett. 114, 181105 (2019). <https://doi.org/10.1063/1.5094838>
- Abe, R., Takeda, T., Shiratori, R., Shirakawa, S., Saito, S. & Baba, T., "Optimization of an H0 photonic crystal nanocavity using machine learning", Opt. Lett. 45, 319-322 (2020). <https://doi.org/10.1364/OL.381616>
- Cakmak, S., Astudillo, R., Frazier, P. & Zhou, E., "Bayesian Optimization of Risk Measures", NeurIPS 33 (2020); arXiv:2007.05554. <https://arxiv.org/abs/2007.05554>
- Hotte-Kilburn, A., "Topological photonic crystal ring resonators", M.Sc. thesis, Concordia University (2021). <https://spectrum.library.concordia.ca/id/eprint/987925/>
- Vasco, J. P. & Savona, V., "Global optimization of an encapsulated Si/SiO2 L3 cavity with a 43 million quality factor", Sci. Rep. 11, 10121 (2021). <https://www.nature.com/articles/s41598-021-89410-1>
- Xie, X., Dang, J., Yan, S., Zhang, W., Hao, H., Xiao, S. et al. (Xu group), "Optimization and robustness of the topological corner state in second-order topological photonic crystals", Opt. Express 29 (2021). <https://doi.org/10.1364/OE.438474>
- Liu, L., Ma, C., Ye, M., Yu, Z., Xue, W. et al., "Photonic Crystal Nanobeam Cavity With a High Experimental Q Factor Exceeding Two Million Based on Machine Learning", J. Lightwave Technol. 40(21), 7150 (2022). <https://opg.optica.org/jlt/abstract.cfm?uri=jlt-40-21-7150>
- Daulton, S., Cakmak, S., Balandat, M., Osborne, M. A., Zhou, E. & Bakshy, E., "Robust Multi-Objective Bayesian Optimization Under Input Noise", ICML 2022, PMLR 162:4831-4866. <https://proceedings.mlr.press/v162/daulton22a.html>
- Albrechtsen, M., Vosoughi Lahijani, B., Christiansen, R. E. et al. (Sigmund, Mørk, Stobbe), "Nanometer-scale photon confinement in topology-optimized dielectric cavities", Nat. Commun. 13, 6281 (2022). <https://doi.org/10.1038/s41467-022-33874-w>
- Liu, L., Long, Y., Fu, K., Zhao, P., Hu, C. et al., "High-Q Silicon Photonic Crystal Ring Resonator Based on Machine Learning", J. Lightwave Technol. (2024). <https://doi.org/10.1109/JLT.2024.3454953>
- Plock, M., Binkowski, F., Zschiedrich, L., Schneider, P.-I. & Burger, S., "Fabrication uncertainty guided design optimization of a photonic crystal cavity by using Gaussian processes", J. Opt. Soc. Am. B 41 (2024); arXiv:2402.09420. <https://doi.org/10.1364/JOSAB.505767>
- Gu, (first author) et al., "Selecting robust silicon photonic designs after Bayesian optimization without extra simulations", Opt. Express 32(21), 37585 (2024). <https://doi.org/10.1364/OE.531213>
- Kim, J., Li, M., Li, Y., Gómez, A., Hinder, O. et al., "Multi-BOWS: multi-fidelity multi-objective Bayesian optimization with warm starts for nanophotonic structure design", Digital Discovery 3, 381 (2024). <https://doi.org/10.1039/D3DD00177F>
- Yu, Z. & Hao, R., "Inverse design of High-Q topological corner states nanocavities based on deep reinforcement learning", Opt. Commun. 577, 131402 (2025). <https://doi.org/10.1016/j.optcom.2024.131402>
- Deep Learning-Optimized, Fabrication Error-Tolerant Photonic Crystal Nanobeam Cavities for Scalable On-Chip Diamond Quantum Systems, arXiv:2502.03987 (2025). <https://arxiv.org/abs/2502.03987>
- Elbek, P., Christiansen, R. E., Aage, N. & Sigmund, O., "Tailoring robust photonic components using stochastic topology optimization", J. Opt. Soc. Am. B (2026). <https://doi.org/10.1364/JOSAB.584921>

### E. Simulation methods

- M. Heiblum, J. H. Harris, 'Analysis of curved optical waveguides by conformal transformation', IEEE J. Quantum Electron. 11(2), 75-83 (1975); correction 12(5), 313 (1976). <https://doi.org/10.1109/JQE.1975.1068563>
- V. A. Mandelshtam, H. S. Taylor, 'Harmonic inversion of time signals and its applications', J. Chem. Phys. 107, 6756-6769 (1997); erratum 109, 4128 (1998). <https://doi.org/10.1063/1.475324> *(not confirmed by the search agents)*
- S. G. Johnson, J. D. Joannopoulos, 'Block-iterative frequency-domain methods for Maxwell's equations in a planewave basis', Optics Express 8(3), 173-190 (2001). <https://doi.org/10.1364/OE.8.000173>
- L. C. Andreani, D. Gerace, 'Photonic-crystal slabs with a triangular lattice of triangular holes investigated using a guided-mode expansion method', Phys. Rev. B 73, 235114 (2006). <https://doi.org/10.1103/PhysRevB.73.235114>
- M. W. McCutcheon, M. Lončar, 'Design of a silicon nitride photonic crystal nanocavity with a Quality factor of one million for coupling to a diamond nanocrystal', Optics Express 16(23), 19136-19145 (2008). <https://doi.org/10.1364/OE.16.019136>
- D. Goldring, U. Levy, I. E. Dotan, A. Tsukernik, M. Oksman, I. Rubin, et al., 'Experimental measurement of quality factor enhancement using slow light modes in one dimensional photonic crystal', Optics Express 16(8), 5585 (2008). <https://doi.org/10.1364/OE.16.005585>
- A. F. Oskooi, D. Roundy, M. Ibanescu, P. Bermel, J. D. Joannopoulos, S. G. Johnson, 'MEEP: A flexible free-software package for electromagnetic simulations by the FDTD method', Computer Physics Communications 181, 687-702 (2010). <https://doi.org/10.1016/j.cpc.2009.11.008>
- M. Khan, T. Babinec, M. W. McCutcheon, P. Deotare, M. Lončar, 'Fabrication and characterization of high-quality-factor silicon nitride nanobeam cavities', Optics Letters 36(3), 421-423 (2011). <https://doi.org/10.1364/OL.36.000421>
- M. Xiao, Z. Q. Zhang, C. T. Chan, 'Surface impedance and bulk band geometric phases in one-dimensional systems', Phys. Rev. X 4, 021017 (2014). <https://doi.org/10.1103/PhysRevX.4.021017>
- G. Gao, Y. Zhang, H. Zhang, Y. Wang, Q. Huang, J. Xia, 'Air-mode photonic crystal ring resonator on silicon-on-insulator', Scientific Reports 6, 19999 (2016). <https://doi.org/10.1038/srep19999>
- P. Lalanne, W. Yan, K. Vynck, C. Sauvan, J.-P. Hugonin, 'Light interaction with photonic and plasmonic resonances', Laser & Photonics Reviews 12(5), 1700113 (2018). <https://doi.org/10.1002/lpor.201700113>
- T. K. Fryett, Y. Chen, J. Whitehead, Z. M. Peycke, X. Xu, A. Majumdar, 'Encapsulated silicon nitride nanobeam cavity for hybrid nanophotonics', ACS Photonics 5, 2176 (2018). <https://arxiv.org/abs/1709.02032>
- K. McGarvey-Lechable, P. Bianucci, 'Bloch-Floquet waves in optical ring resonators', Phys. Rev. B 97, 214204 (2018). <https://arxiv.org/abs/1803.11312>
- T. W. Hughes, I. A. D. Williamson, M. Minkov, S. Fan, 'Forward-mode differentiation of Maxwell's equations', ACS Photonics 6(11), 3010-3016 (2019). <https://doi.org/10.1021/acsphotonics.9b01238>
- M. Blanco de Paz, C. Devescovi, G. Giedke, J. J. Saenz, M. G. Vergniory, B. Bradlyn, D. Bercioux, A. García-Etxarri, 'Tutorial: Computing topological invariants in two-dimensional photonic crystals', Adv. Quantum Technol. 3, 1900117 (2020). <https://doi.org/10.1002/qute.201900117>
- J. Zhan, Z. Jafari, S. Veilleux, M. Dagenais, I. De Leon, 'High-Q nanobeam cavities on a silicon nitride platform enabled by slow light', APL Photonics 5, 066101 (2020). <https://doaj.org/article/5be3b78ba5734cf89bac982fdf6875fd>
- A. M. Hammond, A. Oskooi, M. Chen, Z. Lin, S. G. Johnson, S. E. Ralph, 'High-performance hybrid time/frequency-domain topology optimization for large-scale photonics inverse design', Optics Express 30(3), 4467-4491 (2022). <https://doi.org/10.1364/OE.442074>
- X. Lu, A. Chanana, F. Zhou, M. Davanco, K. Srinivasan, 'Kerr optical parametric oscillation in a photonic crystal microring for accessing the infrared', Optics Letters 47, 3331-3334 (2022). <https://doi.org/10.1364/OL.462494>
- X. Lu, F. Zhou, Y. Sun, A. Chanana, M. Wang, A. McClung, V. Aksyuk, M. Davanço, K. Srinivasan, 'Rod and slit photonic crystal microrings for on-chip cavity quantum electrodynamics', Nanophotonics 12, 521- (2023). <https://arxiv.org/abs/2210.16436>
- 'Photonic-crystal-based high-performance ring resonator using a topological interface state: design and analysis', Applied Optics 62(16), 4281 (2023). <https://doi.org/10.1364/AO.486832>
- S. Zanotti, M. Minkov, D. Nigro, D. Gerace, S. Fan, L. C. Andreani, 'Legume: A free implementation of the guided-mode expansion method for photonic crystal slabs', Computer Physics Communications 304, 109286 (2024). <https://data.mendeley.com/datasets/kf3cwknx4d/1>

### F. 2023-2026 novelty sweep

- A. Hotte-Kilburn, "Topological photonic crystal ring resonators", M.Sc. thesis, Concordia University (2021). <https://spectrum.library.concordia.ca/987925/1/Hotte-Kilburn_Msc_S2021.pdf>
- E. Lucas, S.-P. Yu, T. C. Briles, D. R. Carlson, S. B. Papp, "Tailoring microcombs with inverse-designed, meta-dispersion microresonators", Nature Photonics 17 (2023). <https://www.nature.com/articles/s41566-023-01252-7>
- Q. Xin, R. Li, D. Wei, Z. Zhang, "Filling the simulation-to-reality gap: high-degree-of-freedom AI-optimized photonic crystal nanobeam resonators with fabrication tolerance", Optics Letters 49(15), 4401 (2024). <https://opg.optica.org/ol/abstract.cfm?uri=ol-49-15-4401>
- A. Abulnaga, S. Karg, S. Mukherjee, A. Gupta, K. W. Baldwin, L. N. Pfeiffer, N. P. de Leon, "Design and fabrication of robust hybrid photonic crystal cavities", arXiv:2409.16883 (2024). <https://arxiv.org/abs/2409.16883>
- P. Manfredi, A. Waqas, D. Melati, "Stochastic and multi-objective design of photonic devices with machine learning", Scientific Reports 14 (2024). <https://doi.org/10.1038/s41598-024-57315-4>
- S. Zanotti, M. Minkov, D. Nigro, D. Gerace, S. Fan, L. C. Andreani, "Legume: A free implementation of the guided-mode expansion method for photonic crystal slabs", Computer Physics Communications (2024); also M. Minkov et al., "Inverse design of photonic crystals through automatic differentiation", arXiv:2003.00379 (2020). <https://www.sciencedirect.com/science/article/pii/S0010465524002091>
- E. Lucas et al., "The bandgap-detuned excitation regime in photonic-crystal resonators", arXiv:2404.11733 (2024). <https://arxiv.org/abs/2404.11733>
- D. Kumar, J. K. Rakshit, C. Nayak, M. Hossain, "Photonic crystal-based high performance cascaded micro ring resonator using topological interface state: design, analysis, and application", Optical and Quantum Electronics (2024). <https://doi.org/10.1007/s11082-024-07273-y>
- A. Begovic et al., "BOSON-1: Understanding and enabling physically-robust photonic inverse design with adaptive variation-aware subspace optimization", arXiv:2411.08210 (2024). <https://arxiv.org/abs/2411.08210>
- M. Scherrer, C.-W. Lee, H. Schmid, K. E. Moselund, "Single-mode laser in the telecom range by deterministic amplification of the topological interface mode", ACS Photonics (2024). <https://doi.org/10.1021/acsphotonics.3c01372>
- M.-D. Liu, H.-H. Chen, Z. Wang, Y. Zhang, X. Zhou, G. Tang, et al., "On-chip topological photonic crystal nanobeam filters", Nano Letters (2024). <https://doi.org/10.1021/acs.nanolett.3c04363>
- S. van Haagen, S. Nur, R. Ishihara, "Deep learning-optimized, fabrication error-tolerant photonic crystal nanobeam cavities for scalable on-chip diamond quantum systems", ACS Applied Optical Materials 3(4), 998 (2025). <https://pubs.acs.org/aaoma6/article/3/4/998/3760507/Deep-Learning-Optimized-Fabrication-Error-Tolerant>
- NIST/Papp-group authors (full author list not confirmed), "Implementing photonic-crystal resonator frequency combs in a photonic foundry", Optics Letters (2025). <https://doi.org/10.1364/OL.553980>
- Y. Hu, M. Gao, W. Xie, Y. Xiong, C. Dong, Z. Lu, Y. Lu, S. Assawaworrarit, A. Y. Song, "Absence of universal backscattering immunity in reciprocal photonic waveguides", arXiv:2610.02758 (2026). <https://arxiv.org/abs/2610.02758>
- N. Granchi, C. Gonzini, M. Lodde, G. Calusi, R. P. J. van Veldhoven, A. Fiore, et al., "Improved disorder resilience in small-footprint photonic cavities with periodicity breaking", Advanced Optical Materials (2026). <https://doi.org/10.1002/adom.202502531>
- J. Yu, "Reliability-aware Bayesian optimization of 1310-nm PCSELs with FDTD verification", arXiv:2607.21772 (2026). <https://arxiv.org/abs/2607.21772>
- H. Zhou, H. Yang, N. W. Gangi, T. Xu, R. Huang, J. Gu, "Toward fabrication-robust photonic inverse design: from AI-assisted inverse lithography to yield-driven device optimization", Proc. SPIE (2026). <https://doi.org/10.1117/12.3114569>

### G. Additional works found by the novelty referees

- A. Hotte-Kilburn & P. Bianucci, 'Implementation of the SSH model in an optical ring resonator', J. Opt. 26, 065006 (2024). <https://iopscience.iop.org/article/10.1088/2040-8986/ad42af>
- A. Hotte-Kilburn & P. Bianucci, 'Integrated topological photonics in one dimension', Advances in Physics: X 10, 2476417 (2025). <https://www.tandfonline.com/doi/full/10.1080/23746149.2025.2476417>
- X. Lu, A. McClung, K. Srinivasan, 'High-Q slow light and its localization in a photonic crystal microring', Nat. Photon. 16, 66 (2022). <https://www.nature.com/articles/s41566-021-00912-w>
- K. McGarvey-Lechable et al., 'Slow light in mass-produced, dispersion-engineered photonic crystal ring resonators', Opt. Express 25, 3916 (2017). <https://pubmed.ncbi.nlm.nih.gov/28241601/>
- Y. Gong, L. Guo, S. Wong, A. J. Bennett, S. S. Oh, 'Tailoring topological edge states with photonic crystal nanobeam cavities', Sci. Rep. 11, 1055 (2021). <https://www.nature.com/articles/s41598-020-79915-6>
- Akahane et al., 'Nanocavity Confinement by Orthogonal Valley- and SSH-Topological Interfaces in Glide-Symmetric Photonic Crystal Structures', arXiv:2608.16138 (2026). <https://arxiv.org/html/2608.16138>
- Rosiek et al., 'Observation of strong backscattering in valley-Hall photonic topological interface modes', Nat. Photon. (2023). <http://www.npg.nature.com/articles/s41566-023-01189-x.pdf>
- Gao et al., Sci. Rep. 6, 19999 (2016), 3D FDTD of a uniform hole PhC ring. *(not confirmed by the search agents)*
- Z. Liu, X. Zhan, D. Chen, P. Li, J.-J. Xiao, 'Sublattice-polarized bound state in the continuum in a silicon waveguide array with approximate chiral symmetry', Phys. Rev. A 114, 043537 (published 30 Oct 2026), DOI 10.1103/rsqn-bfgl. <https://journals.aps.org/pra/abstract/10.1103/rsqn-bfgl>
- A. Hotte-Kilburn, P. Bianucci, 'Implementation of the SSH model in an optical ring resonator', J. Opt. (2024), DOI 10.1088/2040-8986/ad42af. <https://beta.iopscience.iop.org/article/10.1088/2040-8986/ad42af>
- M. Proctor, M. Blanco de Paz, D. Bercioux, A. Garcia-Etxarri, P. Arroyo Huidobro, 'Robustness of topological corner modes in photonic crystals', Phys. Rev. Research 2, 042038 (2020), arXiv:2007.10624. <https://par.nsf.gov/servlets/purl/10252406>
- B. Perez-Gonzalez, M. Bello, A. Gomez-Leon, G. Platero, 'Interplay between long-range hopping and disorder in topological systems', PRB 99, 035146 (2019), arXiv:1810.02779; and 'SSH model with long-range hoppings: topology, driving and disorder', arXiv:1802.03973. <https://arxiv.org/html/1810.02779>
- A. Abulnaga et al., 'Design and fabrication of robust hybrid photonic crystal cavities', Nanophotonics (2025, online Nov 2024), DOI 10.1515/nanoph-2024-0500. <https://onlinelibrary.wiley.com/doi/10.1515/nanoph-2024-0500>
- I. Rousseau et al., 'Quantification of scattering loss of III-nitride photonic crystal cavities in the blue spectral range', PRB 95, 125313 (2017). <https://link.aps.org/doi/10.1103/PhysRevB.95.125313>
- M.-D. Liu, J.-W. Dong, X.-D. Chen et al., 'On-Chip Topological Photonic Crystal Nanobeam Filters', Nano Lett. 24, 1635-1641 (2024), DOI 10.1021/acs.nanolett.3c04363. *(not confirmed by the search agents)*
- A. B. Khanikaev, A. Alu, 'Topological photonics: robustness and beyond', Nat. Commun. 15, 931 (2024), DOI 10.1038/s41467-024-45194-2 (the candidate cites it as 2024 but not as Nature Photonics; it is in Nat. Commun.). <https://www.nature.com/articles/s41467-024-45194-2.pdf>
- C. Han, M. Lee, S. Callard, C. Seassal, H. Jeon, 'Lasing at topological edge states in a photonic crystal L3 nanocavity dimer array', Light Sci. Appl. 8, 40 (2019). <https://www.nature.com/articles/s41377-019-0149-7>
- Rosiek, C. A. et al., 'Observation of strong backscattering in valley-Hall photonic topological interface modes', Nat. Photon. 17, 386-392 (2023). <https://www.nature.com/articles/s41566-023-01189-x>
- Arregui, G., Gomis-Bresco, J., Sotomayor-Torres, C. M. & Garcia, P. D., 'Quantifying the Robustness of Topological Slow Light', Phys. Rev. Lett. 126, 027403 (2021). <https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.027403>
- Leykam, D., Xue, H., Zhang, B. & Chong, Y. D., 'Limitations and possibilities of topological photonics', Nat. Rev. Phys. (2025). <https://www.nature.com/articles/s42254-025-00889-3>
- Lee, K. Y. et al., 'Topological guided-mode resonances at non-Hermitian nanophotonic interfaces', Nanophotonics 10 (2021). <https://www.degruyterbrill.com/document/doi/10.1515/nanoph-2021-0024/html>
- Quan, Q. & Loncar, M., 'Deterministic design of wavelength scale, ultra-high Q photonic crystal nanobeam cavities', Opt. Express 19, 18529 (2011). <https://arxiv.org/abs/1108.2675>
- Sanchez-Sanchez, A. et al., 'Topological photonic cavities based on dissimilar Bragg gratings', arXiv:2609.18362 (2026). <https://arxiv.org/abs/2609.18362>
- Wang, H.-X., Li, W., Hu, J. & Jiang, J.-H., 'Photonic Dirac cavity on a gradient dislocation', Front. Phys. (2025/2026), DOI 10.15302/frontphys.2026.032302. <https://doi.org/10.15302/frontphys.2026.032302>
- Gorlach/Barczyk-type QSHE study: 'Interplay of leakage radiation and protection in topological photonic crystal cavities', arXiv:2202.07620 (2022). <https://arxiv.org/abs/2202.07620> *(not confirmed by the search agents)*
- M. Wang, F. Zhou, A. McClung, X. Lu, V. Aksyuk, K. Srinivasan, 'Fractional optical angular momentum and multi-defect-mediated mode re-normalization and orientation control in photonic crystal microring resonators', PRL 129, 186101 (2022); arXiv:2202.09731. <https://arxiv.org/abs/2202.09731>
- X. Lu et al., 'Rod and slit photonic crystal microrings for on-chip cavity quantum electrodynamics' (arXiv:2210.16436; published version on PMC). <https://pmc.ncbi.nlm.nih.gov/articles/PMC11501185/>
- 'Coupled topological interface states', Phys. Rev. B 103, 085412 (2021). <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.103.085412>
- A. A. Majeed & J. Barvestani, 'Coupling of topological interface states in one-dimensional photonic crystals is controlled by spatial distance and cap-layer symmetry', Sci. Rep. (2026). <https://www.nature.com/articles/s41598-026-63584-y>
- Vakulenko, A. et al., 'Adiabatic topological photonic interfaces', Nat. Commun. 14, 4629 (2023). <https://www.nature.com/articles/s41467-023-40238-5>
- Takahashi & Baba, CMA-ES optimization of H0 nanocavities (IEEE Photonics J., 2022). <https://ieeexplore.ieee.org/document/9759999>
- van Haagen, S., Nur, S., Ishihara, R., 'Deep Learning-Optimized, Fabrication Error-Tolerant Photonic Crystal Nanobeam Cavities for Scalable On-Chip Diamond Quantum Systems', ACS Appl. Opt. Mater. 3, 998 (2025). <https://doi.org/10.1021/acsaom.5c00060>
- Kim, J. et al., 'Multi-BOWS: multi-fidelity multi-objective Bayesian optimization with warm starts for nanophotonic structure design', Digital Discovery (2024). <https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00177f>
- Deng, K. et al., 'Terahertz high-quality-factor topological corner-state cavity designed via policy-based reinforcement learning', Phys. Scr. (2026). <https://doi.org/10.1088/1402-4896/aeb1fe>
- A. Sanchez-Sanchez et al. (Malaga/C2N/NRC), 'Topological silicon photonic cavities based on Zak phase engineering of Bragg gratings', EOSAM 2024, Naples (conference). *(not confirmed by the search agents)*
- S. A. Randerson, P. Bouteyre, X. Hu, O. J. Palma Chaundler et al., 'Topological Jackiw-Rebbi states in photonic Van der Waals heterostructures', Light Sci. Appl. (published 2026-07-19). <https://www.nature.com/articles/s41377-026-02392-5>
- K. Y. Lee, K. W. Yoo, Y. Choi, G. Kim, S. Cheon, J. W. Yoon et al., 'Topological guided-mode resonances at non-Hermitian nanophotonic interfaces', Nanophotonics 10, 1853 (2021); and Y. S. Choi et al., 'Topological beaming of light: proof-of-concept experiment', Light Sci. Appl. (2025). <https://doi.org/10.1515/nanoph-2021-0024>
- Experimental demonstration of single mode-splitting in microring with Bragg gratings (Zhejiang Univ., IEEE Photon. Technol. Lett. ~2015), including a pi-phase-shifted inner-wall DBR microring on SOI. <http://coer.zju.edu.cn/download/2015/Experimental%20Demonstration%20of%20Single%20Mode-Splitting%20in%20Microring%20With%20Bragg%20Gratings.pdf>
- X. Lu, Y. Sun, A. Chanana, U. A. Javid, M. Davanco, K. Srinivasan, 'Multi-mode microcavity frequency engineering through a shifted grating in a photonic crystal ring', Photon. Res. 11, A72 (2023). <https://doi.org/10.1364/prj.500375>
- M. Xiao, Z. Q. Zhang, C. T. Chan, 'Surface impedance and bulk band geometric phases in one-dimensional systems', PRX 4, 021017 (2014). *(not confirmed by the search agents)*