#!/usr/bin/env python3
"""
Compare results from different parameter space explorations.
"""

import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

def load_exploration_results():
    """Load results from all exploration runs."""
    results = {}
    
    # Find all results directories
    result_dirs = glob("results/run_*")
    
    for result_dir in result_dirs:
        # Load optimization log
        log_path = os.path.join(result_dir, 'optimization_log.csv')
        best_params_path = os.path.join(result_dir, 'best_params.yaml')
        
        if not os.path.exists(log_path) or not os.path.exists(best_params_path):
            continue
            
        # Load data
        df = pd.read_csv(log_path)
        with open(best_params_path, 'r') as f:
            best_params = yaml.safe_load(f)
        
        # Try to identify which exploration this is based on parameters
        if len(df) < 10:
            continue
            
        # Analyze parameter ranges to classify
        R_range = df['R'].max() - df['R'].min()
        a_range = df['a'].max() - df['a'].min()
        r_min = df['r'].min()
        
        if R_range > 8:  # Large rings
            exploration_type = "Large Rings"
        elif df['R'].max() < 13:  # Compact
            exploration_type = "Compact Designs"
        elif a_range > 0.15:  # Extreme dimerization
            exploration_type = "Extreme Dimerization"
        elif r_min < 0.07:  # Fabrication limits
            exploration_type = "Fabrication Limits"
        elif len(df) > 50:  # Original long run
            exploration_type = "Original Mock Run"
        else:
            exploration_type = "MEEP Test"
        
        results[exploration_type] = {
            'df': df,
            'best_params': best_params,
            'best_score': df['score'].max(),
            'run_dir': result_dir
        }
    
    return results

def create_comparison_report(results):
    """Create a comprehensive comparison report."""
    
    print("=" * 80)
    print("PARAMETER EXPLORATION COMPARISON REPORT")
    print("=" * 80)
    print()
    
    # Summary table
    print("EXPLORATION SUMMARY:")
    print("-" * 60)
    print(f"{'Exploration Type':<25} {'Best Score':<12} {'Iterations':<12} {'Best a/b Ratio':<15}")
    print("-" * 60)
    
    for exp_type, data in results.items():
        best_params = data['best_params']
        dimerization_ratio = best_params['a'] / best_params['b'] if best_params['b'] > 0 else float('inf')
        
        print(f"{exp_type:<25} {data['best_score']:<12.1f} {len(data['df']):<12} {dimerization_ratio:<15.2f}")
    
    print()
    print("DETAILED ANALYSIS:")
    print("-" * 60)
    
    for exp_type, data in results.items():
        df = data['df']
        best_params = data['best_params']
        
        print(f"\n{exp_type.upper()}:")
        print(f"  Best Score: {data['best_score']:.1f}")
        print(f"  Best Parameters:")
        print(f"    a = {best_params['a']:.3f} μm")
        print(f"    b = {best_params['b']:.3f} μm") 
        print(f"    r = {best_params['r']:.3f} μm")
        print(f"    R = {best_params['R']:.3f} μm")
        print(f"    w = {best_params['w']:.3f} μm")
        
        # Calculate key metrics
        dimerization = best_params['a'] / best_params['b']
        circumference = 2 * np.pi * best_params['R']
        unit_cell_length = best_params['a'] + best_params['b']
        num_holes = int(circumference / unit_cell_length) * 2
        
        print(f"  Key Metrics:")
        print(f"    Dimerization ratio (a/b): {dimerization:.2f}")
        print(f"    Estimated holes: {num_holes}")
        print(f"    Ring circumference: {circumference:.1f} μm")
        print(f"    Min feature size: {min(best_params['r'], best_params['a'], best_params['b']):.3f} μm")
        
        # Parameter ranges explored
        print(f"  Parameter Ranges Explored:")
        for param in ['a', 'b', 'r', 'R', 'w']:
            param_min = df[param].min()
            param_max = df[param].max()
            print(f"    {param}: [{param_min:.3f}, {param_max:.3f}] μm")

def plot_exploration_comparison(results):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Best scores
    explorations = list(results.keys())
    scores = [results[exp]['best_score'] for exp in explorations]
    
    axes[0,0].bar(range(len(explorations)), scores, color='skyblue', alpha=0.7)
    axes[0,0].set_xlabel('Exploration Type')
    axes[0,0].set_ylabel('Best Score')
    axes[0,0].set_title('Best Scores by Exploration')
    axes[0,0].set_xticks(range(len(explorations)))
    axes[0,0].set_xticklabels(explorations, rotation=45, ha='right')
    
    # Plot 2: Dimerization ratios
    dimerization_ratios = []
    for exp in explorations:
        best_params = results[exp]['best_params']
        ratio = best_params['a'] / best_params['b'] if best_params['b'] > 0 else float('inf')
        dimerization_ratios.append(min(ratio, 10))  # Cap at 10 for plotting
    
    axes[0,1].bar(range(len(explorations)), dimerization_ratios, color='lightgreen', alpha=0.7)
    axes[0,1].set_xlabel('Exploration Type')
    axes[0,1].set_ylabel('Dimerization Ratio (a/b)')
    axes[0,1].set_title('Dimerization Ratios')
    axes[0,1].set_xticks(range(len(explorations)))
    axes[0,1].set_xticklabels(explorations, rotation=45, ha='right')
    
    # Plot 3: Ring sizes
    ring_sizes = [results[exp]['best_params']['R'] for exp in explorations]
    
    axes[0,2].bar(range(len(explorations)), ring_sizes, color='coral', alpha=0.7)
    axes[0,2].set_xlabel('Exploration Type')
    axes[0,2].set_ylabel('Ring Radius (μm)')
    axes[0,2].set_title('Optimal Ring Sizes')
    axes[0,2].set_xticks(range(len(explorations)))
    axes[0,2].set_xticklabels(explorations, rotation=45, ha='right')
    
    # Plot 4: Hole sizes
    hole_sizes = [results[exp]['best_params']['r'] for exp in explorations]
    
    axes[1,0].bar(range(len(explorations)), hole_sizes, color='gold', alpha=0.7)
    axes[1,0].set_xlabel('Exploration Type')
    axes[1,0].set_ylabel('Hole Radius (μm)')
    axes[1,0].set_title('Optimal Hole Sizes')
    axes[1,0].set_xticks(range(len(explorations)))
    axes[1,0].set_xticklabels(explorations, rotation=45, ha='right')
    
    # Plot 5: Convergence comparison (first few explorations)
    for i, (exp_type, data) in enumerate(list(results.items())[:4]):
        df = data['df']
        if len(df) > 5:  # Only plot if enough data
            cummax_scores = df['score'].cummax()
            axes[1,1].plot(cummax_scores.values, label=exp_type, alpha=0.8, linewidth=2)
    
    axes[1,1].set_xlabel('Iteration')
    axes[1,1].set_ylabel('Best Score So Far')
    axes[1,1].set_title('Convergence Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Plot 6: Parameter space visualization (a vs b)
    colors = plt.cm.tab10(np.linspace(0, 1, len(explorations)))
    
    for i, (exp_type, data) in enumerate(results.items()):
        best_params = data['best_params']
        axes[1,2].scatter(best_params['a'], best_params['b'], 
                         color=colors[i], s=100, alpha=0.8, label=exp_type)
    
    axes[1,2].set_xlabel('a (μm)')
    axes[1,2].set_ylabel('b (μm)')
    axes[1,2].set_title('Optimal (a,b) Points')
    axes[1,2].legend()
    axes[1,2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_exploration_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    print("Loading exploration results...")
    results = load_exploration_results()
    
    if not results:
        print("No exploration results found!")
        return
    
    print(f"Found {len(results)} different explorations")
    print()
    
    # Create text report
    create_comparison_report(results)
    
    # Create plots
    print("\nGenerating comparison plots...")
    plot_exploration_comparison(results)
    
    print(f"\nComparison plot saved as: parameter_exploration_comparison.png")

if __name__ == "__main__":
    main()