"""
Tests for simulation wrapper functions
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.simulation_wrapper import (
    evaluate_design_mock,
    evaluate_design_meep,
    _generate_ssh_ring_geometry,
    _calculate_objective,
    _simulate_physics_model,
)


class TestSimulationWrapper:
    """Test suite for simulation wrapper functions"""

    @pytest.fixture
    def basic_config(self):
        """Provide a basic configuration for testing"""
        return {
            'simulation': {
                'resolution': 40,
                'pml_width': 2.0,
                'sim_time': 200,
                'target_wavelength': 1.547,
            },
            'objective': {
                'num_disorder_runs': 5,
                'disorder_std_dev_percent': 5.0,
                'q_penalty_factor': 2.0,
            }
        }

    @pytest.fixture
    def valid_design(self):
        """Provide a valid design vector"""
        return [0.35, 0.15, 0.14, 12.0, 0.50]  # [a, b, r, R, w]

    def test_evaluate_design_mock_basic(self, valid_design, basic_config):
        """Test that mock evaluation runs without errors"""
        score = evaluate_design_mock(valid_design, basic_config)
        assert isinstance(score, (int, float))
        assert score > -1e9  # Should not be NEGINF for valid design

    def test_evaluate_design_meep_basic(self, valid_design, basic_config):
        """Test that MEEP evaluation runs (with physics mock)"""
        score = evaluate_design_meep(valid_design, basic_config)
        assert isinstance(score, (int, float))
        assert score > -1e9  # Should not be NEGINF for valid design

    def test_generate_ssh_ring_geometry(self, valid_design):
        """Test SSH ring geometry generation"""
        a, b, r, R, w = valid_design
        holes, num_cells = _generate_ssh_ring_geometry(a, b, r, R, w, disorder_std=0.0)

        assert len(holes) > 0
        assert num_cells > 0
        assert len(holes) == num_cells * 2  # Two holes per unit cell

        # Check hole structure
        for x, y, hole_r in holes:
            assert isinstance(x, float)
            assert isinstance(y, float)
            assert hole_r > 0

    def test_calculate_objective_valid(self, basic_config):
        """Test objective calculation with valid Q-factors"""
        q_factors = [10000, 12000, 11000, 10500, 11500]
        score = _calculate_objective(q_factors, basic_config)

        q_avg = np.mean(q_factors)
        q_std = np.std(q_factors)
        expected = q_avg - 2.0 * q_std

        assert abs(score - expected) < 1e-6

    def test_calculate_objective_empty(self, basic_config):
        """Test objective calculation with no Q-factors"""
        q_factors = []
        score = _calculate_objective(q_factors, basic_config)
        assert score == -1.0e10  # Should be NEGINF

    def test_simulate_physics_model(self, valid_design):
        """Test physics-based model simulation"""
        a, b, r, R, w = valid_design
        holes, _ = _generate_ssh_ring_geometry(a, b, r, R, w)

        q = _simulate_physics_model(a, b, r, R, w, holes)
        assert isinstance(q, (int, float))
        assert q >= 1000  # Minimum Q enforced

    def test_disorder_runs_consistency(self, valid_design, basic_config):
        """Test that multiple evaluations produce consistent results"""
        scores = []
        for _ in range(3):
            score = evaluate_design_mock(valid_design, basic_config)
            scores.append(score)

        # Scores should be similar but not identical due to randomness
        assert all(isinstance(s, (int, float)) for s in scores)
        assert np.std(scores) > 0  # Some variation expected

    def test_invalid_design_handling(self, basic_config):
        """Test handling of invalid design parameters"""
        # Design with negative values
        invalid_design = [-0.1, 0.15, 0.14, 12.0, 0.50]

        # Should still run but may produce low score
        score = evaluate_design_mock(invalid_design, basic_config)
        assert isinstance(score, (int, float))

    def test_comprehensive_objectives(self, valid_design, basic_config):
        """Test comprehensive objective calculation"""
        basic_config['return_comprehensive_objectives'] = True
        result = evaluate_design_mock(valid_design, basic_config)

        assert isinstance(result, dict)
        assert 'q_factor' in result
        assert 'q_std' in result
        assert 'bandgap_size' in result
        assert 'mode_volume' in result
        assert 'score' in result

    def test_geometry_with_disorder(self, valid_design):
        """Test geometry generation with disorder"""
        a, b, r, R, w = valid_design
        disorder_std = 0.01

        holes1, _ = _generate_ssh_ring_geometry(a, b, r, R, w, disorder_std)
        holes2, _ = _generate_ssh_ring_geometry(a, b, r, R, w, disorder_std)

        # Same number of holes
        assert len(holes1) == len(holes2)

        # But different radii due to disorder
        radii1 = [h[2] for h in holes1]
        radii2 = [h[2] for h in holes2]
        assert radii1 != radii2  # Should be different due to randomness
