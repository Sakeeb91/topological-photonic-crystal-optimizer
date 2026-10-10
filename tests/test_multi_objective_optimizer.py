"""
Tests for the multi-objective optimization problem setup
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.multi_objective_optimizer import TopologicalPhotonicCrystalProblem


@pytest.fixture
def mo_config():
    return {
        'design_space': {
            'a': [0.30, 0.60],
            'b': [0.05, 0.20],
            'r': [0.03, 0.07],
            'w': [0.40, 0.70],
            'N_cells': [80, 150],
            'coupling_gap': [0.1, 0.5],
            'coupling_width': [0.3, 0.8],
        },
        'objective': {'num_disorder_runs': 2},
        'optimizer': {'algorithm': 'NSGA3', 'population_size': 4, 'n_generations': 1},
    }


def test_constraints_min_feature_size_is_used(mo_config):
    mo_config['constraints'] = {'min_feature_size': 0.08}
    problem = TopologicalPhotonicCrystalProblem(mo_config, simulation_function=None)
    assert problem.constraints.min_feature_size == 0.08


def test_default_min_feature_size(mo_config):
    problem = TopologicalPhotonicCrystalProblem(mo_config, simulation_function=None)
    assert problem.constraints.min_feature_size == 0.05


def _stub_simulation(design_vector, config):
    return {'q_factor': 30000.0, 'q_std': 500.0, 'bandgap_size': 10.0, 'mode_volume': 0.1}


def test_evaluate_reports_constraints_to_pymoo(mo_config):
    """Feasible designs get G <= 0 and infeasible ones G > 0"""
    import numpy as np

    problem = TopologicalPhotonicCrystalProblem(mo_config, _stub_simulation)
    assert problem.n_ieq_constr == 2
    #               a     b     r     w     N    gap  width
    X = np.array([[0.40, 0.18, 0.04, 0.50, 100, 0.2, 0.5],    # b - 2r = 0.10 > 0.05
                  [0.40, 0.10, 0.04, 0.50, 100, 0.2, 0.5]])   # b - 2r = 0.02 < 0.05
    out = {}
    problem._evaluate(X, out)
    assert (out['G'][0] <= 0).all()
    assert out['G'][1][0] > 0
    assert out['F'][0][0] == -30000.0  # feasible design was simulated


def test_empty_pareto_front_when_nothing_feasible(mo_config):
    from types import SimpleNamespace
    from src.multi_objective_optimizer import MultiObjectiveOptimizer

    optimizer = MultiObjectiveOptimizer(mo_config, _stub_simulation)
    pareto_df = optimizer.analyze_pareto_front(SimpleNamespace(X=None, F=None))
    assert pareto_df.empty
