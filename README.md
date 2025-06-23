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

1. Switch to mock simulations in `run_optimization.py`:
   ```python
   from src.simulation_wrapper import evaluate_design_mock as evaluate_design
   # from src.simulation_wrapper import evaluate_design_meep as evaluate_design
   ```
2. Run optimization: `python run_optimization.py --config configs/strong_dimerization_v1.yaml`
3. Results will be saved in `results/run_TIMESTAMP/`
4. Analyze results: `python src/analysis.py results/run_TIMESTAMP`

## MEEP Electromagnetic Simulations

The framework now includes a complete MEEP implementation:

1. **Current Status**: MEEP simulation structure is implemented with physics-based modeling
2. **To enable full MEEP**: Install MEEP and uncomment the MEEP import lines in `src/simulation_wrapper.py`
3. **Run MEEP simulations**: 
   ```bash
   python run_optimization.py --config configs/meep_production_v1.yaml
   ```
4. **Test quickly**: Use `configs/test_meep_v1.yaml` for rapid testing

## Visualization and Analysis

Analyze and visualize your results:

```bash
# Generate analysis report
python src/analysis.py results/run_TIMESTAMP

# Visualize best design geometry
python visualize_best_design.py results/run_TIMESTAMP
```

## Configuration Files

Multiple configurations are available:

- `strong_dimerization_v1.yaml` - Original mock simulation config
- `test_meep_v1.yaml` - Fast MEEP testing (5 iterations, 5 disorder runs)
- `meep_production_v1.yaml` - Full production MEEP config (150 iterations, 10 disorder runs)

Each config includes:
- **Design space**: Parameter bounds for optimization  
- **Simulation settings**: MEEP resolution, materials, boundary conditions
- **Objective function**: Disorder runs, penalty factors, quality filters
- **Optimizer settings**: Bayesian optimization parameters
- **Fabrication constraints**: Minimum feature sizes, aspect ratios

## Results

Each optimization run creates a timestamped directory in `results/` containing:
- `optimization_log.csv`: Complete parameter and score history
- `best_params.yaml`: Best parameters found  
- `run_config.yaml`: Configuration used for reproducibility
- `analysis_report.md`: Statistical analysis and correlations
- `optimization_plots.png`: Progress visualization
- `best_design_geometry.png`: Geometric visualization of optimal design
- `best_design_geometry_report.md`: Detailed geometry analysis

## Key Features

✅ **Complete MEEP Integration**: Full electromagnetic simulation workflow  
✅ **Bayesian Optimization**: Intelligent parameter space exploration  
✅ **Disorder Robustness**: Multiple simulations with fabrication errors  
✅ **Geometry Analysis**: Automated design validation and visualization  
✅ **Reproducible Research**: All parameters and results tracked  
✅ **Configurable**: Easy switching between mock and real simulations  

## Physics Background

This optimizer designs topological photonic crystal ring resonators with:
- **Parameters**: `a` (dimerization 1), `b` (dimerization 2), `r` (hole radius), `R` (ring radius), `w` (width)
- **Objective**: Maximize Q-factor while ensuring robustness to fabrication disorder
- **Method**: Bayesian optimization with Gaussian Process surrogate model
- **Evaluation**: Multiple FDTD simulations with random hole radius perturbations