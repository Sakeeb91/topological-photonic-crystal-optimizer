# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning optimization project for designing topological photonic crystal ring resonators. The goal is to maximize the Q-factor (quality factor) of photonic cavities while ensuring robustness to fabrication errors.

**Key Physics Concepts:**
- Photonic crystals: Structures that control light propagation using periodic patterns
- Ring resonators: Circular waveguides that trap light at specific wavelengths  
- Q-factor: Measure of how well light is confined (higher = better)
- Topological protection: Design robustness to manufacturing imperfections
- SSH (Su-Schrieffer-Heeger) model: Creates topological edge states via dimerization

## Project Structure (To Be Implemented)

The project follows this planned architecture:

```
topological-optimizer/
├── configs/                    # YAML configuration files
│   └── strong_dimerization_v1.yaml
├── results/                    # Optimization results and logs
├── src/                        # Core implementation modules
│   ├── simulation_wrapper.py   # Interface to MEEP FDTD simulator
│   ├── analysis.py            # Result analysis and visualization
│   └── utils.py               # Utility functions
├── run_optimization.py        # Main optimization script
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Design Parameters

The optimization searches over 5 geometric parameters:
- `a`: First dimerization distance (typically 0.30-0.40 μm)
- `b`: Second dimerization distance (typically 0.10-0.20 μm)  
- `r`: Hole radius (typically 0.10-0.18 μm)
- `R`: Ring radius (typically 10.0-15.0 μm)
- `w`: Waveguide width (typically 0.45-0.55 μm)

## Core Technologies

- **MEEP**: Finite-difference time-domain (FDTD) electromagnetic simulator
- **scikit-optimize**: Bayesian optimization for parameter search
- **Harminv**: Signal processing tool within MEEP for extracting Q-factors
- **NumPy/Pandas**: Data manipulation and analysis

## Key Implementation Notes

**Objective Function:**
- Must evaluate robustness by running multiple simulations with disorder
- Score = Q_average - penalty_factor × Q_standard_deviation
- Each evaluation requires ~10 FDTD simulations with random hole radius perturbations

**Simulation Workflow:**
1. Generate ring geometry with holes placed via trigonometry
2. Add random disorder to hole radii (typically 5% std deviation)
3. Run MEEP FDTD simulation with Harminv mode analysis
4. Extract Q-factor and wavelength from resonant modes
5. Repeat for disorder averaging
6. Calculate final robustness score

**Bayesian Optimization:**
- Uses Gaussian Process surrogate model
- Balances exploration vs exploitation via acquisition functions
- Typically requires 20 initial random samples + 100 optimization iterations
- Each iteration selects most promising design to evaluate next

## Development Approach

**Testing Strategy:**
- Start with mock simulation functions for rapid prototyping
- Mock functions use simple analytical models to validate optimization loop
- Replace with real MEEP simulations once framework is validated

**Configuration Management:**
- All parameters defined in YAML config files
- Separate physics parameters, simulation settings, and optimization hyperparameters
- Results stored with timestamp and config for reproducibility

## Common Commands

Since this is a research project without existing build infrastructure:

```bash
# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Run optimization
python run_optimization.py --config configs/strong_dimerization_v1.yaml

# Install MEEP (separate installation required)
# Follow official MEEP documentation for platform-specific instructions
```

## Architecture Principles

**Separation of Concerns:**
- `simulation_wrapper.py`: Physics simulation interface
- `run_optimization.py`: ML optimization orchestration  
- Configuration files: Parameter management
- Results directory: Data persistence and analysis

**Modularity:**
- Easily switch between mock and real simulation functions
- Pluggable objective functions for different optimization goals
- Configurable optimizer settings without code changes

**Scientific Reproducibility:**
- All runs logged with parameters and results
- Random seeds controlled for deterministic optimization
- Configuration files stored with results for exact reproduction