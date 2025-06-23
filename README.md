# Topological Photonic Crystal Optimizer

This project uses Bayesian Optimization to find the optimal geometry for a topological photonic crystal ring resonator, as described in the thesis by A. Hotte-Kilburn.

## Goal
Maximize the disorder-robust Q-factor of the edge-state cavity.

## Setup
1. Install MEEP and its Python interface. Follow the official instructions.
2. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run Optimization:**
   ```bash
   python run_optimization.py --config configs/strong_dimerization_v1.yaml
   ```

2. **Analyze Results:**
   ```bash
   python src/analysis.py results/run_YYYYMMDD_HHMMSS
   ```

## Project Structure

```
topological-optimizer/
├── configs/                    # Configuration files
│   └── strong_dimerization_v1.yaml
├── results/                    # Optimization results (auto-generated)
├── src/                        # Core modules
│   ├── simulation_wrapper.py   # MEEP interface
│   ├── analysis.py            # Result analysis
│   └── utils.py               # Utilities
├── run_optimization.py        # Main script
├── requirements.txt           # Dependencies
├── README.md                  # This file
└── CLAUDE.md                  # AI assistant guidance
```

## Quick Start with Mock Simulations

To test the optimization framework without MEEP:

1. The default configuration uses mock simulations for rapid testing
2. Run optimization: `python run_optimization.py --config configs/strong_dimerization_v1.yaml`
3. Results will be saved in `results/run_TIMESTAMP/`
4. Analyze results: `python src/analysis.py results/run_TIMESTAMP`

## Switching to Real MEEP Simulations

1. Install MEEP following official documentation
2. Edit `run_optimization.py`:
   ```python
   # Change this line:
   from src.simulation_wrapper import evaluate_design_mock as evaluate_design
   # To this:
   from src.simulation_wrapper import evaluate_design_meep as evaluate_design
   ```
3. Implement the MEEP simulation logic in `src/simulation_wrapper.py` function `evaluate_design_meep()`

## Configuration

Edit `configs/strong_dimerization_v1.yaml` to modify:
- **Design space**: Parameter bounds for optimization
- **Simulation settings**: MEEP resolution, simulation time
- **Objective function**: Disorder runs, penalty factors
- **Optimizer settings**: Number of iterations, acquisition function

## Results

Each optimization run creates a timestamped directory in `results/` containing:
- `optimization_log.csv`: Complete parameter and score history
- `best_params.yaml`: Best parameters found
- `run_config.yaml`: Configuration used for reproducibility
- `analysis_report.md`: Auto-generated analysis (when using analysis.py)
- `optimization_plots.png`: Progress visualization

## Physics Background

This optimizer designs topological photonic crystal ring resonators with:
- **Parameters**: `a` (dimerization 1), `b` (dimerization 2), `r` (hole radius), `R` (ring radius), `w` (width)
- **Objective**: Maximize Q-factor while ensuring robustness to fabrication disorder
- **Method**: Bayesian optimization with Gaussian Process surrogate model
- **Evaluation**: Multiple FDTD simulations with random hole radius perturbations