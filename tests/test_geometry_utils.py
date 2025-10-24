"""
Tests for geometry utility functions
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.geometry_utils import (
    analyze_geometry_properties,
    validate_geometry_constraints,
)


class TestGeometryUtils:
    """Test suite for geometry utility functions"""

    @pytest.fixture
    def valid_design(self):
        """Provide a valid design vector"""
        return [0.35, 0.15, 0.14, 12.0, 0.50]  # [a, b, r, R, w]

    @pytest.fixture
    def basic_config(self):
        """Provide basic configuration"""
        return {
            'fabrication': {
                'min_feature_size': 0.05,
                'max_aspect_ratio': 10.0,
            }
        }

    def test_analyze_geometry_properties(self, valid_design):
        """Test geometry property analysis"""
        analysis = analyze_geometry_properties(valid_design)

        assert isinstance(analysis, dict)
        assert 'dimerization_ratio' in analysis
        assert 'unit_cell_length' in analysis
        assert 'num_unit_cells' in analysis
        assert 'total_holes' in analysis
        assert 'filling_factor' in analysis
        assert 'min_feature_size' in analysis

        # Check reasonable values
        a, b, r, R, w = valid_design
        assert analysis['dimerization_ratio'] == pytest.approx(a / b)
        assert analysis['unit_cell_length'] == pytest.approx(a + b)
        assert analysis['total_holes'] > 0

    def test_validate_geometry_constraints_valid(self, valid_design, basic_config):
        """Test constraint validation with valid design"""
        violations = validate_geometry_constraints(valid_design, basic_config)
        assert isinstance(violations, list)

    def test_validate_geometry_constraints_too_small_feature(self, basic_config):
        """Test detection of too-small features"""
        # Design with very small hole radius
        invalid_design = [0.35, 0.15, 0.02, 12.0, 0.50]
        violations = validate_geometry_constraints(invalid_design, basic_config)

        assert len(violations) > 0
        assert any('radius' in v.lower() for v in violations)

    def test_validate_geometry_constraints_no_dimerization(self, basic_config):
        """Test detection of no dimerization"""
        # Design where a <= b
        invalid_design = [0.15, 0.35, 0.14, 12.0, 0.50]
        violations = validate_geometry_constraints(invalid_design, basic_config)

        assert len(violations) > 0
        assert any('dimerization' in v.lower() for v in violations)

    def test_validate_geometry_constraints_hole_too_large(self, basic_config):
        """Test detection of hole larger than waveguide"""
        # Design where 2*r >= w
        invalid_design = [0.35, 0.15, 0.26, 12.0, 0.50]
        violations = validate_geometry_constraints(invalid_design, basic_config)

        assert len(violations) > 0
        assert any('diameter' in v.lower() or 'width' in v.lower() for v in violations)

    def test_geometry_properties_edge_cases(self):
        """Test geometry analysis with edge case parameters"""
        # Very small ring
        small_ring = [0.30, 0.10, 0.10, 5.0, 0.45]
        analysis = analyze_geometry_properties(small_ring)
        assert analysis['num_unit_cells'] > 0

        # Very large ring
        large_ring = [0.35, 0.15, 0.14, 50.0, 0.50]
        analysis = analyze_geometry_properties(large_ring)
        assert analysis['num_unit_cells'] > 0
        assert analysis['total_holes'] > 100  # Should have many holes

    def test_filling_factor_calculation(self):
        """Test filling factor calculation"""
        design = [0.35, 0.15, 0.14, 12.0, 0.50]
        analysis = analyze_geometry_properties(design)

        # Filling factor should be between 0 and 1
        assert 0 <= analysis['filling_factor'] <= 1

    def test_validate_with_empty_config(self, valid_design):
        """Test validation with minimal config"""
        minimal_config = {}
        violations = validate_geometry_constraints(valid_design, minimal_config)
        assert isinstance(violations, list)
