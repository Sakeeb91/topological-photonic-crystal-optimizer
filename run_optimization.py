import os
import sys
import yaml
import argparse
import time
from datetime import datetime
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
from tqdm import tqdm

# Import our wrapper function
# We can easily switch between the mock and real one here!
# from src.simulation_wrapper import evaluate_design_mock as evaluate_design
from src.simulation_wrapper import evaluate_design_meep as evaluate_design

# Import utility functions
from src.utils import validate_config, save_config_with_timestamp

# --- 1. Setup ---
def setup_directories(run_name):
    """Creates a unique directory for storing results of this run."""
    results_dir = os.path.join("results", run_name)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

def load_config(config_path):
    """Loads the YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- 2. Optimizer Definition ---
def define_search_space(config):
    """Defines the search space for the optimizer from the config file."""
    space = []
    param_names = []
    for name, bounds in config['design_space'].items():
        space.append(Real(low=bounds[0], high=bounds[1], name=name))
        param_names.append(name)
    return space, param_names

def main(config_path):
    # Load config and setup
    config = load_config(config_path)

    # Validate configuration before starting optimization
    try:
        validate_config(config)
        print("✓ Configuration validated successfully")
    except ValueError as e:
        print(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir = setup_directories(run_name)
    print(f"Starting optimization run: {run_name}")
    print(f"Results will be saved in: {results_dir}")

    # Save configuration with timestamp for reproducibility
    save_config_with_timestamp(config, results_dir)

    space, param_names = define_search_space(config)
    
    # We create a progress bar for the optimization
    pbar = tqdm(total=config['optimizer']['n_initial_points'] + config['optimizer']['n_iterations'])

    # --- 3. The Objective Function (What the Optimizer Calls) ---
    # The @use_named_args decorator converts a list of parameters to keyword arguments
    @use_named_args(space)
    def objective_function(**params):
        design_vector = [params[name] for name in param_names]
        
        # The optimizer wants to MINIMIZE, so we return the NEGATIVE of our score
        score = evaluate_design(design_vector, config)
        pbar.update(1)
        
        # Log progress
        log_data = {name: [val] for name, val in params.items()}
        log_data['score'] = [-score] # Store the real score, not the negative
        log_df = pd.DataFrame(log_data)
        
        log_path = os.path.join(results_dir, 'optimization_log.csv')
        log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)
        
        return -score

    # --- 4. Run the Optimizer ---
    print("\nStarting Bayesian Optimization...")
    result = gp_minimize(
        func=objective_function,
        dimensions=space,
        n_calls=config['optimizer']['n_initial_points'] + config['optimizer']['n_iterations'],
        n_initial_points=config['optimizer']['n_initial_points'],
        acq_func=config['optimizer']['acquisition_function'],
        random_state=123 # for reproducibility
    )
    pbar.close()

    # --- 5. Save and Print Final Results ---
    print("\nOptimization finished!")
    best_params = {name: val for name, val in zip(param_names, result.x)}
    best_score = -result.fun

    print(f"Best score found: {best_score:.4f}")
    print("Best parameters:")
    for name, val in best_params.items():
        print(f"  {name}: {val:.4f}")

    # Save best parameters to a file for analysis
    with open(os.path.join(results_dir, 'best_params.yaml'), 'w') as f:
        yaml.dump(best_params, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization for Topological Photonic Crystals.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    main(args.config)