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
