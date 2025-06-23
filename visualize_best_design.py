#!/usr/bin/env python3
"""
Script to visualize the best design found by optimization.
"""

import yaml
import sys
import os
from src.geometry_utils import create_geometry_report, visualize_ring_geometry

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_best_design.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    # Load best parameters
    best_params_path = os.path.join(results_dir, 'best_params.yaml')
    if not os.path.exists(best_params_path):
        print(f"No best_params.yaml found in {results_dir}")
        sys.exit(1)
    
    with open(best_params_path, 'r') as f:
        best_params = yaml.safe_load(f)
    
    # Convert to design vector
    design_vector = [
        best_params['a'],
        best_params['b'], 
        best_params['r'],
        best_params['R'],
        best_params['w']
    ]
    
    # Load configuration (try to find the config used)
    config_path = os.path.join(results_dir, 'run_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default config
        config = {
            'simulation': {'resolution': 40, 'pml_width': 2.0},
            'fabrication': {'min_feature_size': 0.05, 'max_aspect_ratio': 10.0}
        }
    
    print("Best Design Parameters:")
    print(f"  a = {best_params['a']:.4f} μm")
    print(f"  b = {best_params['b']:.4f} μm")
    print(f"  r = {best_params['r']:.4f} μm")
    print(f"  R = {best_params['R']:.4f} μm")
    print(f"  w = {best_params['w']:.4f} μm")
    print()
    
    # Create geometry report and visualization
    output_path = os.path.join(results_dir, 'best_design_geometry.png')
    
    print("Creating geometry visualization...")
    report_text, analysis, violations = create_geometry_report(
        design_vector, config, save_path=output_path
    )
    
    print("Geometry Analysis:")
    print(f"  Dimerization ratio (a/b): {analysis['dimerization_ratio']:.2f}")
    print(f"  Total holes: {analysis['total_holes']}")
    print(f"  Unit cells: {analysis['num_unit_cells']:.1f}")
    print(f"  Filling factor: {analysis['filling_factor']:.3f}")
    print(f"  Min feature size: {analysis['min_feature_size']:.3f} μm")
    
    if violations:
        print("\nConstraint Violations:")
        for violation in violations:
            print(f"  - {violation}")
    else:
        print("\n✓ All fabrication constraints satisfied")
    
    print(f"\nVisualization saved to: {output_path}")
    print(f"Report saved to: {output_path.replace('.png', '_report.md')}")

if __name__ == "__main__":
    main()