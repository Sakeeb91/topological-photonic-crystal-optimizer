#!/usr/bin/env python3
"""
Multi-Objective Optimization for Topological Photonic Crystal Ring Resonators

This script implements the advanced ML optimization framework based on insights
from the AlexisHK thesis, addressing the fundamental trade-offs in topological
photonic crystal design.

Key Features:
- NSGA-III multi-objective optimization
- Physics-informed constraints
- Extended parameter space with discrete variables
- Enhanced disorder robustness modeling
"""

import os
import sys
import argparse
import yaml
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from multi_objective_optimizer import MultiObjectiveOptimizer
from simulation_wrapper import evaluate_design_mock

def setup_directories(run_name):
    """Create directories for storing results."""
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different types of results
    os.makedirs(os.path.join(results_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "designs"), exist_ok=True)
    
    return results_dir

def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_trade_off_analysis_plots(pareto_df, output_dir):
    """
    Create comprehensive trade-off analysis plots based on thesis insights.
    """
    plt.style.use('default')
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (15, 12)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Q-factor vs Q-factor std (robustness trade-off)
    scatter = axes[0,0].scatter(pareto_df['q_avg'], pareto_df['q_std'], 
                               c=pareto_df['dimerization_ratio'], 
                               cmap='viridis', alpha=0.7, s=60)
    axes[0,0].set_xlabel('Average Q-factor')
    axes[0,0].set_ylabel('Q-factor Standard Deviation')
    axes[0,0].set_title('Performance vs Robustness Trade-off')
    cbar1 = plt.colorbar(scatter, ax=axes[0,0])
    cbar1.set_label('Dimerization Ratio (a/b)')
    
    # Plot 2: Q-factor vs Bandgap (fundamental physics trade-off from thesis)
    scatter2 = axes[0,1].scatter(pareto_df['bandgap_size'], pareto_df['q_avg'],
                                c=pareto_df['R'], cmap='plasma', alpha=0.7, s=60)
    axes[0,1].set_xlabel('Bandgap Size')
    axes[0,1].set_ylabel('Average Q-factor')
    axes[0,1].set_title('Bandgap vs Q-factor Trade-off\n(Lattice vs Radiation Confinement)')
    cbar2 = plt.colorbar(scatter2, ax=axes[0,1])
    cbar2.set_label('Ring Radius (μm)')
    
    # Plot 3: Mode Volume vs Q-factor (Purcell factor analysis)
    scatter3 = axes[0,2].scatter(pareto_df['mode_volume'], pareto_df['q_avg'],
                                c=pareto_df['r'], cmap='coolwarm', alpha=0.7, s=60)
    axes[0,2].set_xlabel('Mode Volume')
    axes[0,2].set_ylabel('Average Q-factor')
    axes[0,2].set_title('Mode Volume vs Q-factor\n(Purcell Factor Optimization)')
    cbar3 = plt.colorbar(scatter3, ax=axes[0,2])
    cbar3.set_label('Hole Radius (μm)')
    
    # Plot 4: Dimerization Strength Analysis
    axes[1,0].scatter(pareto_df['dimerization_ratio'], pareto_df['q_robustness'],
                     c=pareto_df['N_cells'], cmap='magma', alpha=0.7, s=60)
    axes[1,0].set_xlabel('Dimerization Ratio (a/b)')
    axes[1,0].set_ylabel('Robustness Ratio (Q_avg/Q_std)')
    axes[1,0].set_title('Dimerization vs Robustness')
    axes[1,0].axvline(x=3, color='red', linestyle='--', alpha=0.5, label='Thesis "Strong" Design')
    axes[1,0].legend()
    
    # Plot 5: Ring Size vs Performance
    axes[1,1].scatter(pareto_df['R'], pareto_df['q_avg'],
                     c=pareto_df['bandgap_size'], cmap='Spectral', alpha=0.7, s=60)
    axes[1,1].set_xlabel('Ring Radius (μm)')
    axes[1,1].set_ylabel('Average Q-factor')
    axes[1,1].set_title('Ring Size vs Performance')
    
    # Plot 6: Multi-dimensional parameter correlation
    # Create a custom score that balances multiple objectives
    pareto_df['composite_score'] = (pareto_df['q_avg'] / pareto_df['q_avg'].max() + 
                                   pareto_df['bandgap_size'] / pareto_df['bandgap_size'].max() -
                                   pareto_df['q_std'] / pareto_df['q_std'].max() -
                                   pareto_df['mode_volume'] / pareto_df['mode_volume'].max())
    
    top_designs = pareto_df.nlargest(10, 'composite_score')
    
    # Parameter parallel coordinates plot
    params_to_plot = ['a', 'b', 'r', 'R', 'w']
    normalized_params = top_designs[params_to_plot].copy()
    for param in params_to_plot:
        normalized_params[param] = (normalized_params[param] - normalized_params[param].min()) / \
                                  (normalized_params[param].max() - normalized_params[param].min())
    
    for i, (idx, row) in enumerate(normalized_params.iterrows()):
        axes[1,2].plot(range(len(params_to_plot)), row.values, 
                      marker='o', alpha=0.7, linewidth=2, label=f'Design {i+1}')
    
    axes[1,2].set_xticks(range(len(params_to_plot)))
    axes[1,2].set_xticklabels(params_to_plot)
    axes[1,2].set_ylabel('Normalized Parameter Value')
    axes[1,2].set_title('Top 10 Design Parameter Profiles')
    axes[1,2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_objective_trade_off_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return top_designs

def generate_design_recommendations(pareto_df, output_dir):
    """
    Generate design recommendations for different application scenarios.
    """
    recommendations = {}
    
    # 1. Maximum Q-factor design
    max_q_design = pareto_df.loc[pareto_df['q_avg'].idxmax()]
    recommendations['maximum_q_factor'] = {
        'description': 'Highest Q-factor for ultra-sensitive applications',
        'parameters': max_q_design.to_dict(),
        'application': 'Biological sensing, frequency references'
    }
    
    # 2. Most robust design (highest Q/std ratio)
    most_robust_design = pareto_df.loc[pareto_df['q_robustness'].idxmax()]
    recommendations['maximum_robustness'] = {
        'description': 'Most fabrication-tolerant design',
        'parameters': most_robust_design.to_dict(),
        'application': 'Commercial manufacturing, mass production'
    }
    
    # 3. Largest bandgap design
    max_bandgap_design = pareto_df.loc[pareto_df['bandgap_size'].idxmax()]
    recommendations['maximum_topological_protection'] = {
        'description': 'Strongest topological protection',
        'parameters': max_bandgap_design.to_dict(),
        'application': 'Research into topological phenomena'
    }
    
    # 4. Smallest mode volume design  
    min_mode_vol_design = pareto_df.loc[pareto_df['mode_volume'].idxmin()]
    recommendations['minimum_mode_volume'] = {
        'description': 'Tightest light confinement (highest Purcell factor)',
        'parameters': min_mode_vol_design.to_dict(),
        'application': 'Quantum optics, single-photon sources'
    }
    
    # 5. Balanced design (composite score)
    pareto_df['composite_score'] = (pareto_df['q_avg'] / pareto_df['q_avg'].max() + 
                                   pareto_df['bandgap_size'] / pareto_df['bandgap_size'].max() -
                                   pareto_df['q_std'] / pareto_df['q_std'].max() -
                                   pareto_df['mode_volume'] / pareto_df['mode_volume'].max())
    balanced_design = pareto_df.loc[pareto_df['composite_score'].idxmax()]
    recommendations['balanced_performance'] = {
        'description': 'Best overall balance of all objectives',
        'parameters': balanced_design.to_dict(),
        'application': 'General-purpose applications, telecommunications'
    }
    
    # Save recommendations
    with open(os.path.join(output_dir, 'design_recommendations.yaml'), 'w') as f:
        yaml.dump(recommendations, f, default_flow_style=False)
    
    # Create summary report
    report_lines = [
        "# Multi-Objective Optimization Design Recommendations",
        "",
        "Based on Pareto optimal solutions from multi-objective optimization.",
        "",
    ]
    
    for rec_type, rec_data in recommendations.items():
        report_lines.extend([
            f"## {rec_type.replace('_', ' ').title()}",
            "",
            f"**Description**: {rec_data['description']}",
            f"**Application**: {rec_data['application']}",
            "",
            "**Optimal Parameters**:",
        ])
        
        params = rec_data['parameters']
        report_lines.extend([
            f"- a = {params['a']:.3f} μm (primary dimerization)",
            f"- b = {params['b']:.3f} μm (secondary dimerization)", 
            f"- r = {params['r']:.3f} μm (hole radius)",
            f"- R = {params['R']:.3f} μm (ring radius)",
            f"- w = {params['w']:.3f} μm (waveguide width)",
            f"- N_cells = {params['N_cells']:.0f} (unit cells)",
            "",
            "**Performance Metrics**:",
            f"- Q-factor: {params['q_avg']:.0f} ± {params['q_std']:.0f}",
            f"- Bandgap size: {params['bandgap_size']:.3f}",
            f"- Mode volume: {params['mode_volume']:.4f}",
            f"- Dimerization ratio: {params['dimerization_ratio']:.2f}",
            f"- Robustness ratio: {params['q_robustness']:.1f}",
            "",
        ])
    
    with open(os.path.join(output_dir, 'design_recommendations.md'), 'w') as f:
        f.write('\n'.join(report_lines))
    
    return recommendations

def main():
    """Main optimization workflow."""
    parser = argparse.ArgumentParser(description="Multi-Objective Optimization for Topological Photonic Crystals")
    parser.add_argument('--config', type=str, default='configs/multi_objective_v1.yaml',
                       help='Configuration file path')
    parser.add_argument('--generations', type=int, default=None,
                       help='Number of generations (overrides config)')
    parser.add_argument('--population-size', type=int, default=None,
                       help='Population size (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.generations:
        config['optimizer']['n_generations'] = args.generations
    if args.population_size:
        config['optimizer']['population_size'] = args.population_size
    
    # Setup output directory
    if args.output_dir:
        results_dir = args.output_dir
        os.makedirs(results_dir, exist_ok=True)
    else:
        run_name = f"multi_obj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir = setup_directories(run_name)
    
    print(f"Results will be saved to: {results_dir}")
    
    # Display optimization settings
    print("\n" + "="*70)
    print("MULTI-OBJECTIVE OPTIMIZATION FOR TOPOLOGICAL PHOTONIC CRYSTALS")
    print("="*70)
    print(f"Algorithm: {config['optimizer']['algorithm']}")
    print(f"Population size: {config['optimizer']['population_size']}")
    print(f"Generations: {config['optimizer']['n_generations']}")
    print(f"Objectives: Q-factor avg, Q-factor std, Bandgap size, Mode volume")
    print(f"Parameter space: {len(config['design_space'])} dimensions")
    
    # Setup simulation function
    def simulation_wrapper(design_vector, sim_config):
        """Wrapper to ensure correct configuration format."""
        # Merge configs appropriately
        full_config = {**config, **sim_config}
        return evaluate_design_mock(design_vector, full_config)
    
    # Create and run optimizer
    print("\nInitializing multi-objective optimizer...")
    optimizer = MultiObjectiveOptimizer(config, simulation_wrapper)
    
    print("\nStarting optimization...")
    start_time = time.time()
    
    result = optimizer.optimize(config['optimizer']['n_generations'])
    
    end_time = time.time()
    print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    print("\nAnalyzing Pareto front...")
    pareto_df = optimizer.save_results(result, results_dir)
    
    # Generate comprehensive analysis
    print("Creating trade-off analysis plots...")
    top_designs = create_trade_off_analysis_plots(pareto_df, os.path.join(results_dir, "plots"))
    
    print("Generating design recommendations...")
    recommendations = generate_design_recommendations(pareto_df, os.path.join(results_dir, "designs"))
    
    # Print summary
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"Total designs evaluated: {len(optimizer.problem.evaluation_history)}")
    print(f"Pareto optimal solutions: {len(pareto_df)}")
    print(f"Best Q-factor: {pareto_df['q_avg'].max():.0f}")
    print(f"Best robustness ratio: {pareto_df['q_robustness'].max():.1f}")
    print(f"Largest bandgap: {pareto_df['bandgap_size'].max():.3f}")
    print(f"Smallest mode volume: {pareto_df['mode_volume'].min():.4f}")
    
    print(f"\nTop 3 balanced designs:")
    balanced_top3 = pareto_df.nlargest(3, 'composite_score')
    for i, (idx, design) in enumerate(balanced_top3.iterrows()):
        print(f"  {i+1}. a={design['a']:.3f}, b={design['b']:.3f}, r={design['r']:.3f}, "
              f"R={design['R']:.3f}, Q={design['q_avg']:.0f}±{design['q_std']:.0f}")
    
    print(f"\nResults saved to: {results_dir}")
    print("Files generated:")
    print("  - pareto_front.csv: Complete Pareto optimal solutions")
    print("  - evaluation_history.pkl: Full optimization history")
    print("  - plots/multi_objective_trade_off_analysis.png: Trade-off analysis")
    print("  - designs/design_recommendations.yaml: Application-specific designs")
    print("  - designs/design_recommendations.md: Human-readable recommendations")

if __name__ == "__main__":
    main()