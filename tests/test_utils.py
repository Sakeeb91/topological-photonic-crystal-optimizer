"""
Tests for utility functions
"""
import pytest
import sys
import os
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import (
    validate_config,
    create_parameter_summary,
    estimate_num_holes,
    check_fabrication_constraints,
    load_yaml_safe,
)


class TestUtils:
    """Test suite for utility functions"""

    @pytest.fixture
    def valid_config(self):
        """Provide a valid configuration"""
        return {
            'design_space': {
                'a': [0.30, 0.40],
                'b': [0.10, 0.20],
                'r': [0.10, 0.18],
                'R': [10.0, 15.0],
                'w': [0.45, 0.55],
            },
            'simulation': {
                'resolution': 40,
                'pml_width': 2.0,
                'sim_time': 200,
                'target_wavelength': 1.547,
            },
            'objective': {
                'num_disorder_runs': 10,
                'disorder_std_dev_percent': 5.0,
                'q_penalty_factor': 2.0,
            },
            'optimizer': {
                'n_initial_points': 20,
                'n_iterations': 100,
                'acquisition_function': 'gp_hedge',
            }
        }

    def test_validate_config_valid(self, valid_config):
        """Test that valid config passes validation"""
        assert validate_config(valid_config) is True

    def test_validate_config_missing_section(self, valid_config):
        """Test that missing section raises error"""
        del valid_config['design_space']
        with pytest.raises(ValueError, match="Missing required section"):
            validate_config(valid_config)

    def test_validate_config_missing_parameter(self, valid_config):
        """Test that missing parameter raises error"""
        del valid_config['design_space']['a']
        with pytest.raises(ValueError, match="Missing parameter"):
            validate_config(valid_config)

    def test_validate_config_invalid_bounds(self, valid_config):
        """Test that invalid bounds raise error"""
        valid_config['design_space']['a'] = [0.40, 0.30]  # min > max
        with pytest.raises(ValueError, match="min bound must be less than max"):
            validate_config(valid_config)

    def test_create_parameter_summary(self):
        """Test parameter summary creation"""
        design_vector = [0.35, 0.15, 0.14, 12.0, 0.50]
        summary = create_parameter_summary(design_vector)

        assert isinstance(summary, str)
        assert 'a:' in summary
        assert 'b:' in summary
        assert 'Dimerization ratio' in summary

    def test_estimate_num_holes(self):
        """Test hole number estimation"""
        R = 12.0
        a = 0.35
        b = 0.15

        total_holes, num_pairs = estimate_num_holes(R, a, b)

        assert total_holes > 0
        assert num_pairs > 0
        assert total_holes == num_pairs * 2

    def test_check_fabrication_constraints_valid(self):
        """Test fabrication constraints with valid design"""
        design_vector = [0.35, 0.15, 0.14, 12.0, 0.50]
        violations = check_fabrication_constraints(design_vector, min_feature_size=0.05)

        assert isinstance(violations, list)
        # May or may not have violations depending on parameters

    def test_check_fabrication_constraints_invalid(self):
        """Test fabrication constraints with invalid design"""
        # Design with hole larger than waveguide
        design_vector = [0.35, 0.15, 0.30, 12.0, 0.50]  # r=0.30, w=0.50 -> 2r > w
        violations = check_fabrication_constraints(design_vector)

        assert len(violations) > 0
        assert any('diameter' in v.lower() for v in violations)

    def test_load_yaml_safe(self):
        """Test safe YAML loading"""
        # Create temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'test': 'value', 'number': 42}, f)
            temp_path = f.name

        try:
            data = load_yaml_safe(temp_path)
            assert data['test'] == 'value'
            assert data['number'] == 42
        finally:
            os.unlink(temp_path)

    def test_load_yaml_safe_missing_file(self):
        """Test error handling for missing file"""
        with pytest.raises(FileNotFoundError):
            load_yaml_safe('/nonexistent/file.yaml')

    def test_parameter_summary_custom_names(self):
        """Test parameter summary with custom names"""
        design_vector = [0.35, 0.15, 0.14]
        param_names = ['alpha', 'beta', 'radius']
        summary = create_parameter_summary(design_vector, param_names)

        assert 'alpha:' in summary
        assert 'beta:' in summary
        assert 'radius:' in summary
