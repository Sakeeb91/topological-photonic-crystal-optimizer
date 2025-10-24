"""
Topological Photonic Crystal Optimizer

A comprehensive machine learning framework for designing disorder-robust
topological photonic crystal ring resonators with SSH (Su-Schrieffer-Heeger)
model implementation.
"""

__version__ = "1.0.0"
__author__ = "Sakeeb Rahman"

# Core simulation functions
from .simulation_wrapper import (
    evaluate_design_mock,
    evaluate_design_meep,
)

# Analysis and visualization
from .analysis import (
    load_optimization_results,
    plot_optimization_progress,
    analyze_parameter_correlations,
    generate_analysis_report,
)

# Geometry utilities
from .geometry_utils import (
    visualize_ring_geometry,
    analyze_geometry_properties,
    validate_geometry_constraints,
    create_geometry_report,
)

# Utility functions
from .utils import (
    validate_config,
    create_parameter_summary,
    estimate_num_holes,
    check_fabrication_constraints,
    load_yaml_safe,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",

    # Simulation
    "evaluate_design_mock",
    "evaluate_design_meep",

    # Analysis
    "load_optimization_results",
    "plot_optimization_progress",
    "analyze_parameter_correlations",
    "generate_analysis_report",

    # Geometry
    "visualize_ring_geometry",
    "analyze_geometry_properties",
    "validate_geometry_constraints",
    "create_geometry_report",

    # Utils
    "validate_config",
    "create_parameter_summary",
    "estimate_num_holes",
    "check_fabrication_constraints",
    "load_yaml_safe",
]
