import numpy as np
import yaml
import os
from datetime import datetime

def validate_config(config):
    """Validate the configuration file for required fields and reasonable values."""
    required_sections = ['design_space', 'simulation', 'objective', 'optimizer']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate design space
    design_space = config['design_space']
    required_params = ['a', 'b', 'r', 'R', 'w']
    
    for param in required_params:
        if param not in design_space:
            raise ValueError(f"Missing parameter in design_space: {param}")
        
        bounds = design_space[param]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Parameter {param} bounds must be a list of [min, max]")
        
        if bounds[0] >= bounds[1]:
            raise ValueError(f"Parameter {param} min bound must be less than max bound")
    
    # Validate physical constraints
    if design_space['a'][0] <= design_space['b'][1]:
        print("Warning: 'a' parameter range overlaps with 'b' range. "
              "For strong dimerization, 'a' should be significantly larger than 'b'.")
    
    if design_space['r'][1] * 2 >= design_space['w'][0]:
        raise ValueError("Hole diameter (2*r) cannot be larger than waveguide width (w)")
    
    return True

def create_parameter_summary(design_vector, param_names=None):
    """Create a formatted summary of design parameters."""
    if param_names is None:
        param_names = ['a', 'b', 'r', 'R', 'w']
    
    summary = "Design Parameters:\n"
    for i, (name, value) in enumerate(zip(param_names, design_vector)):
        summary += f"  {name}: {value:.4f} μm\n"
    
    # Calculate derived quantities
    a, b, r, R, w = design_vector[:5]
    dimerization_ratio = a / b if b > 0 else float('inf')
    filling_factor = (r**2) / (w**2) if w > 0 else 0
    
    summary += f"\nDerived Quantities:\n"
    summary += f"  Dimerization ratio (a/b): {dimerization_ratio:.2f}\n"
    summary += f"  Hole filling factor: {filling_factor:.3f}\n"
    summary += f"  Ring circumference: {2 * np.pi * R:.1f} μm\n"
    
    return summary

def estimate_num_holes(R, a, b):
    """Estimate the number of holes that fit around the ring."""
    circumference = 2 * np.pi * R
    average_spacing = (a + b) / 2
    num_pairs = int(circumference / (a + b))
    total_holes = num_pairs * 2
    return total_holes, num_pairs

def save_config_with_timestamp(config, results_dir):
    """Save the configuration with a timestamp for reproducibility."""
    config_copy = config.copy()
    config_copy['run_info'] = {
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
    }
    
    config_path = os.path.join(results_dir, 'run_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False)
    
    return config_path

def get_git_commit_hash():
    """Get the current git commit hash for reproducibility."""
    try:
        import subprocess
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def format_time_duration(seconds):
    """Format a time duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

def check_fabrication_constraints(design_vector, min_feature_size=0.05):
    """Check if design parameters meet fabrication constraints."""
    a, b, r, R, w = design_vector[:5]
    
    violations = []
    
    # Minimum feature size check
    if r < min_feature_size:
        violations.append(f"Hole radius {r:.3f} μm below minimum feature size {min_feature_size} μm")
    
    if a < min_feature_size:
        violations.append(f"Spacing 'a' {a:.3f} μm below minimum feature size {min_feature_size} μm")
    
    if b < min_feature_size:
        violations.append(f"Spacing 'b' {b:.3f} μm below minimum feature size {min_feature_size} μm")
    
    # Physical constraints
    if 2 * r >= w:
        violations.append(f"Hole diameter {2*r:.3f} μm >= waveguide width {w:.3f} μm")
    
    # Practical constraints
    if a <= b:
        violations.append(f"Spacing 'a' {a:.3f} μm <= spacing 'b' {b:.3f} μm (no dimerization)")
    
    return violations

class ProgressTracker:
    """Simple progress tracking utility."""
    
    def __init__(self, total_iterations):
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.start_time = datetime.now()
        self.best_score = -np.inf
        
    def update(self, score):
        """Update progress with new score."""
        self.current_iteration += 1
        if score > self.best_score:
            self.best_score = score
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.current_iteration > 0:
            avg_time_per_iter = elapsed / self.current_iteration
            remaining_time = avg_time_per_iter * (self.total_iterations - self.current_iteration)
            
            print(f"Iteration {self.current_iteration}/{self.total_iterations} "
                  f"(Best: {self.best_score:.2f}, "
                  f"ETA: {format_time_duration(remaining_time)})")
    
    def finish(self):
        """Print final summary."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        print(f"\nCompleted {self.current_iteration} iterations in {format_time_duration(elapsed)}")
        print(f"Best score achieved: {self.best_score:.4f}")

def load_yaml_safe(file_path):
    """Safely load a YAML file with error handling."""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def create_results_summary(results_dir):
    """Create a quick summary of results for easy viewing."""
    try:
        from .analysis import load_optimization_results
        df, best_params = load_optimization_results(results_dir)
        
        summary = f"""
Optimization Results Summary
============================

Total evaluations: {len(df)}
Best score: {df['score'].max():.4f}
Final score: {df['score'].iloc[-1]:.4f}
Average score: {df['score'].mean():.4f}

Best parameters:
"""
        if best_params:
            for param, value in best_params.items():
                summary += f"  {param}: {value:.4f}\n"
        
        return summary
        
    except Exception as e:
        return f"Error creating summary: {e}"