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
    validate_multi_objective_config,
    get_min_feature_size,
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

    @pytest.mark.parametrize("seed", [0, 123, None])
    def test_validate_config_accepts_valid_seed(self, valid_config, seed):
        """Non-negative integer or null seeds are accepted"""
        valid_config['seed'] = seed
        assert validate_config(valid_config) is True

    @pytest.mark.parametrize("seed", [-1, 1.5, "42", True])
    def test_validate_config_rejects_invalid_seed(self, valid_config, seed):
        """Negative, non-integer, string, and bool seeds are rejected"""
        valid_config['seed'] = seed
        with pytest.raises(ValueError, match="seed must be"):
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


REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')


class TestValidateMultiObjectiveConfig:
    """Tests for validate_multi_objective_config"""

    @pytest.fixture
    def mo_config(self):
        """Minimal valid multi-objective configuration"""
        return {
            'design_space': {
                'a': [0.30, 0.60],
                'b': [0.05, 0.20],
                'r': [0.05, 0.18],
                'w': [0.40, 0.70],
                'N_cells': [80, 150],
                'coupling_gap': [0.1, 0.5],
                'coupling_width': [0.3, 0.8],
            },
            'objective': {'num_disorder_runs': 8},
            'optimizer': {
                'algorithm': 'NSGA3',
                'population_size': 40,
                'n_generations': 50,
                'n_partitions': 4,
            },
        }

    def test_valid_config(self, mo_config):
        assert validate_multi_objective_config(mo_config) is True

    def test_shipped_multi_objective_config_is_valid(self):
        config = load_yaml_safe(os.path.join(REPO_ROOT, 'configs', 'multi_objective_v1.yaml'))
        assert validate_multi_objective_config(config) is True

    @pytest.mark.parametrize("section", ['design_space', 'objective', 'optimizer'])
    def test_missing_section(self, mo_config, section):
        del mo_config[section]
        with pytest.raises(ValueError, match=f"Missing required section: {section}"):
            validate_multi_objective_config(mo_config)

    @pytest.mark.parametrize("param", ['a', 'b', 'r', 'w', 'N_cells', 'coupling_gap', 'coupling_width'])
    def test_missing_design_parameter(self, mo_config, param):
        del mo_config['design_space'][param]
        with pytest.raises(ValueError, match=f"Missing parameter in design_space: {param}"):
            validate_multi_objective_config(mo_config)

    @pytest.mark.parametrize("bounds, message", [
        ([0.60, 0.30], "min bound must be less than max"),
        ([0.30], "list of two numbers"),
        (["0.3", 0.6], "list of two numbers"),
        ([-0.1, 0.6], "min bound must be positive"),
    ])
    def test_invalid_bounds(self, mo_config, bounds, message):
        mo_config['design_space']['a'] = bounds
        with pytest.raises(ValueError, match=message):
            validate_multi_objective_config(mo_config)

    def test_non_integer_n_cells(self, mo_config):
        mo_config['design_space']['N_cells'] = [80.5, 150]
        with pytest.raises(ValueError, match="N_cells bounds must be positive integers"):
            validate_multi_objective_config(mo_config)

    def test_empty_feasible_region_hole_spacing(self, mo_config):
        # b_max - 2*r_min = 0.20 - 0.16 = 0.04 <= 0.05: every design violates b - 2r > 0.05
        mo_config['design_space']['r'] = [0.08, 0.16]
        with pytest.raises(ValueError, match="No feasible designs: b_max - 2\\*r_min"):
            validate_multi_objective_config(mo_config)

    def test_empty_feasible_region_edge_clearance(self, mo_config):
        mo_config['design_space']['w'] = [0.10, 0.19]
        with pytest.raises(ValueError, match="No feasible designs: \\(w_max"):
            validate_multi_objective_config(mo_config)

    def test_missing_num_disorder_runs(self, mo_config):
        mo_config['objective'] = {'objectives': []}
        with pytest.raises(ValueError, match="num_disorder_runs must be a positive integer"):
            validate_multi_objective_config(mo_config)

    def test_unsupported_algorithm(self, mo_config):
        mo_config['optimizer']['algorithm'] = 'NSGA2'
        with pytest.raises(ValueError, match="must be 'NSGA3'"):
            validate_multi_objective_config(mo_config)

    @pytest.mark.parametrize("key, value", [
        ('population_size', 0),
        ('n_generations', -5),
        ('n_generations', 2.5),
        ('n_partitions', 0),
    ])
    def test_invalid_optimizer_settings(self, mo_config, key, value):
        mo_config['optimizer'][key] = value
        with pytest.raises(ValueError, match=f"optimizer.{key} must be a positive integer"):
            validate_multi_objective_config(mo_config)

    def test_uses_constraints_min_feature_size(self, mo_config):
        """A box feasible at the 0.05 default is rejected under constraints.min_feature_size"""
        mo_config['design_space']['r'] = [0.04, 0.18]  # b_max - 2*r_min = 0.12
        assert validate_multi_objective_config(mo_config) is True
        mo_config['constraints'] = {'min_feature_size': 0.13}
        with pytest.raises(ValueError, match="min_feature_size = 0.13"):
            validate_multi_objective_config(mo_config)

    def test_invalid_seed(self, mo_config):
        mo_config['seed'] = -3
        with pytest.raises(ValueError, match="seed must be"):
            validate_multi_objective_config(mo_config)


class TestGetMinFeatureSize:
    """Tests for get_min_feature_size"""

    def test_default(self):
        assert get_min_feature_size({}) == 0.05

    @pytest.mark.parametrize("config, expected", [
        ({'min_feature_size': 0.06}, 0.06),
        ({'fabrication': {'min_feature_size': 0.07}, 'min_feature_size': 0.06}, 0.07),
        ({'constraints': {'min_feature_size': 0.08},
          'fabrication': {'min_feature_size': 0.07}}, 0.08),
        ({'constraints': {'max_ring_radius': 25.0}, 'fabrication': None}, 0.05),
    ])
    def test_precedence(self, config, expected):
        assert get_min_feature_size(config) == expected
