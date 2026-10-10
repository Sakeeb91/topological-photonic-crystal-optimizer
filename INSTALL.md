# Installation Guide

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) MEEP for full electromagnetic simulations

### 2. Clone the Repository

```bash
git clone https://github.com/Sakeeb91/topological-photonic-crystal-optimizer.git
cd topological-photonic-crystal-optimizer
```

### 3. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows
```

### 4. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install pytest pytest-cov black flake8
```

### 5. Install as Package (Optional)

```bash
# Install in editable mode for development
pip install -e .

# Or install normally
pip install .
```

### 6. Verify Installation

```bash
# Run tests
pytest

# Try importing the package
python -c "import src; print('Import successful!')"

# Run a quick optimization
python run_optimization.py --config configs/strong_dimerization_v1.yaml
```

## Installing MEEP (Optional)

For full electromagnetic simulations, install MEEP:

```bash
# Using conda (recommended)
conda install -c conda-forge pymeep

# Or follow the official MEEP installation guide:
# https://meep.readthedocs.io/en/latest/Installation/
```

After installing MEEP:
1. Uncomment the `import meep as mp` line in `src/simulation_wrapper.py`
2. Set `MEEP_AVAILABLE = True` in the same file
3. Uncomment the MEEP simulation code in `evaluate_design_meep()`

## Troubleshooting

### Common Issues

**Import errors**: Make sure virtual environment is activated
```bash
source venv/bin/activate
```

**Missing dependencies**: Reinstall requirements
```bash
pip install -r requirements.txt --force-reinstall
```

**MEEP not found**: Use conda for MEEP installation, pip installation may not work
```bash
conda install -c conda-forge pymeep
```

### Getting Help

- Check the [README.md](README.md) for usage examples
- Review [CLAUDE.md](CLAUDE.md) for development guidelines
- Open an issue on GitHub for bugs or questions

## Development Setup

For contributing to the project:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks (if using)
# pip install pre-commit
# pre-commit install

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format code
black src/ tests/

# Lint code
flake8 src/ tests/
```
