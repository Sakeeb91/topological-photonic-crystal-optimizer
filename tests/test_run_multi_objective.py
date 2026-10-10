"""
Tests for run_multi_objective_optimization.py helpers
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_multi_objective_optimization import make_simulation_function

DESIGN = [0.35, 0.15, 0.06, 12.0, 0.50]  # [a, b, r, R, w]


def _config(simulation):
    return {
        'simulation': simulation,
        'objective': {'num_disorder_runs': 2, 'q_penalty_factor': 0.5},
        'seed': 0,
    }


@pytest.mark.parametrize("simulation", [
    {'return_comprehensive_objectives': True},
    {},  # defaults to full objectives
])
def test_returns_full_objectives(simulation):
    config = _config(simulation)
    result = make_simulation_function(config)(DESIGN, config)
    assert isinstance(result, dict)
    assert {'q_factor', 'q_std', 'bandgap_size', 'mode_volume'} <= result.keys()


def test_scalar_score_when_disabled():
    config = _config({'return_comprehensive_objectives': False})
    result = make_simulation_function(config)(DESIGN, config)
    assert isinstance(result, float)


def test_setup_directories_creates_subdirectories(tmp_path):
    """--output-dir paths get the plots/ and designs/ subdirectories too"""
    from run_multi_objective_optimization import setup_directories

    results_dir = tmp_path / 'custom_output'
    assert setup_directories(str(results_dir)) == str(results_dir)
    assert (results_dir / 'plots').is_dir()
    assert (results_dir / 'designs').is_dir()
