import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os

def load_optimization_results(results_dir):
    """Load optimization results from a results directory."""
    log_path = os.path.join(results_dir, 'optimization_log.csv')
    best_params_path = os.path.join(results_dir, 'best_params.yaml')
    
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"No optimization log found at {log_path}")
    
    df = pd.read_csv(log_path)
    
    best_params = None
    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            best_params = yaml.safe_load(f)
    
    return df, best_params

def plot_optimization_progress(df, save_path=None):
    """Plot the optimization progress over iterations."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot score vs iteration
    ax1.plot(df.index, df['score'], 'b-', alpha=0.7, label='Score')
    ax1.plot(df.index, df['score'].cummax(), 'r-', linewidth=2, label='Best so far')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Score')
    ax1.set_title('Optimization Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot parameter evolution for key parameters
    param_cols = [col for col in df.columns if col != 'score']
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_cols)))
    
    for i, param in enumerate(param_cols):
        ax2.plot(df.index, df[param], color=colors[i], alpha=0.7, label=param)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('Parameter Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_parameter_correlations(df):
    """Analyze correlations between parameters and objective score."""
    param_cols = [col for col in df.columns if col != 'score']
    
    correlations = {}
    for param in param_cols:
        corr = df[param].corr(df['score'])
        correlations[param] = corr
    
    return correlations

def generate_analysis_report(results_dir, save_report=True):
    """Generate a comprehensive analysis report."""
    df, best_params = load_optimization_results(results_dir)
    
    report = []
    report.append("# Optimization Analysis Report")
    report.append(f"Generated: {pd.Timestamp.now()}")
    report.append(f"Results directory: {results_dir}")
    report.append("")
    
    # Basic statistics
    report.append("## Summary Statistics")
    report.append(f"Total iterations: {len(df)}")
    report.append(f"Best score: {df['score'].max():.4f}")
    report.append(f"Average score: {df['score'].mean():.4f}")
    report.append(f"Score std dev: {df['score'].std():.4f}")
    report.append("")
    
    # Best parameters
    if best_params:
        report.append("## Best Parameters")
        for param, value in best_params.items():
            report.append(f"- {param}: {value:.4f}")
        report.append("")
    
    # Parameter correlations
    correlations = analyze_parameter_correlations(df)
    report.append("## Parameter Correlations with Score")
    for param, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
        report.append(f"- {param}: {corr:.4f}")
    report.append("")
    
    # Convergence analysis
    improvement_iterations = df[df['score'] > df['score'].shift()].index.tolist()
    report.append("## Convergence Analysis")
    report.append(f"Iterations with improvement: {len(improvement_iterations)}")
    if improvement_iterations:
        report.append(f"Last improvement at iteration: {max(improvement_iterations)}")
    report.append("")
    
    report_text = "\n".join(report)
    
    if save_report:
        report_path = os.path.join(results_dir, 'analysis_report.md')
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Also save plots
        plot_path = os.path.join(results_dir, 'optimization_plots.png')
        plot_optimization_progress(df, save_path=plot_path)
        plt.close()
    
    return report_text, df

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analysis.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    try:
        report, df = generate_analysis_report(results_dir)
        print(report)
    except Exception as e:
        print(f"Error analyzing results: {e}")
        sys.exit(1)