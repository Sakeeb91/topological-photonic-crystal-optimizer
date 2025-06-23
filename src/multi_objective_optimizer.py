#!/usr/bin/env python3
"""
Multi-Objective Bayesian Optimization for Topological Photonic Crystal Ring Resonators

This module implements advanced multi-objective optimization based on insights from the 
AlexisHK thesis, specifically addressing the trade-off between lattice confinement 
and radiation confinement.

Key Features:
- NSGA-III multi-objective optimization
- Physics-informed constraint handling
- Extended parameter space with discrete variables
- Enhanced disorder robustness modeling
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import time
from datetime import datetime
import yaml
import os

# Multi-objective optimization
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.core.variable import Real, Integer
from pymoo.core.mixed import MixedVariableGA

# Gaussian Process for surrogate modeling
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler

# Physics-informed modeling
import scipy.optimize as opt
from scipy import constants


@dataclass
class OptimizationObjectives:
    """Container for optimization objectives based on thesis analysis."""
    q_avg: float = 0.0          # Average Q-factor
    q_std: float = 0.0          # Q-factor standard deviation (lower is better)
    bandgap_size: float = 0.0   # Bandgap size (higher is better)
    mode_volume: float = 0.0    # Mode volume (lower is better)
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for optimization algorithms."""
        # Note: We negate q_avg and bandgap_size because pymoo minimizes
        return np.array([-self.q_avg, self.q_std, -self.bandgap_size, self.mode_volume])


@dataclass 
class ExtendedDesignParameters:
    """Extended parameter space based on thesis analysis."""
    # Core geometric parameters
    a: float = 0.35            # Primary dimerization distance (μm)
    b: float = 0.15            # Secondary dimerization distance (μm) 
    r: float = 0.12            # Hole radius (μm)
    w: float = 0.50            # Waveguide width (μm)
    
    # Discrete parameter
    N_cells: int = 100         # Number of unit cells in ring
    
    # Derived parameter (from constraint 2πR = N * (a + b))
    R: float = 0.0            # Ring radius (μm) - calculated automatically
    
    # Experimental coupling parameters
    coupling_gap: float = 0.2  # Gap to coupling waveguide (μm)
    coupling_width: float = 0.5 # Coupling waveguide width (μm)
    
    def __post_init__(self):
        """Calculate ring radius from constraint equation."""
        self.R = self.calculate_ring_radius()
    
    def calculate_ring_radius(self) -> float:
        """Calculate ring radius from 2πR = N * (a + b) constraint."""
        unit_cell_length = self.a + self.b
        return (self.N_cells * unit_cell_length) / (2 * np.pi)
    
    def to_optimization_vector(self) -> np.ndarray:
        """Convert to optimization vector (excluding derived R)."""
        return np.array([self.a, self.b, self.r, self.w, self.N_cells, 
                        self.coupling_gap, self.coupling_width])
    
    @classmethod
    def from_optimization_vector(cls, x: np.ndarray) -> 'ExtendedDesignParameters':
        """Create from optimization vector."""
        params = cls(
            a=x[0], b=x[1], r=x[2], w=x[3], N_cells=int(x[4]),
            coupling_gap=x[5], coupling_width=x[6]
        )
        return params


class PhysicsInformedConstraints:
    """Physics-informed constraints based on thesis analysis."""
    
    def __init__(self, min_feature_size: float = 0.05):
        """
        Initialize constraints.
        
        Args:
            min_feature_size: Minimum fabrication feature size (μm)
        """
        self.min_feature_size = min_feature_size
    
    def check_constraints(self, params: ExtendedDesignParameters) -> Tuple[bool, List[str]]:
        """
        Check all physics-informed constraints.
        
        Returns:
            (is_feasible, violated_constraints)
        """
        violations = []
        
        # Fabrication constraint 1: b - 2r > min_feature_size
        if params.b - 2 * params.r <= self.min_feature_size:
            violations.append(f"Hole spacing too small: {params.b - 2*params.r:.4f} <= {self.min_feature_size}")
        
        # Fabrication constraint 2: (w - 2r)/2 > min_feature_size  
        edge_clearance = (params.w - 2 * params.r) / 2
        if edge_clearance <= self.min_feature_size:
            violations.append(f"Edge clearance too small: {edge_clearance:.4f} <= {self.min_feature_size}")
        
        # Physical constraint: holes must fit within waveguide
        if 2 * params.r >= params.w:
            violations.append(f"Holes too large for waveguide: 2r={2*params.r:.4f} >= w={params.w:.4f}")
        
        # Physical constraint: positive dimerization possible
        if params.a <= 0 or params.b <= 0:
            violations.append("Negative dimerization distances")
        
        # Ring radius constraint (automatically satisfied by construction)
        # 2πR = N * (a + b) is enforced in parameter calculation
        
        return len(violations) == 0, violations
    
    def constraint_penalty(self, params: ExtendedDesignParameters) -> float:
        """Calculate penalty for constraint violations."""
        penalty = 0.0
        
        # Soft penalty for fabrication constraints
        hole_spacing_violation = max(0, self.min_feature_size - (params.b - 2 * params.r))
        edge_clearance_violation = max(0, self.min_feature_size - (params.w - 2 * params.r) / 2)
        
        penalty += 1000 * (hole_spacing_violation + edge_clearance_violation)
        
        # Hard penalty for physical impossibilities
        if 2 * params.r >= params.w:
            penalty += 10000
        if params.a <= 0 or params.b <= 0:
            penalty += 10000
            
        return penalty


class EnhancedDisorderModel:
    """Enhanced disorder modeling with multiple fabrication error types."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize disorder model from configuration."""
        self.hole_radius_std = config.get('hole_radius_disorder_std', 0.05)
        self.sidewall_roughness_std = config.get('sidewall_roughness_std', 0.01)
        self.enable_sidewall_roughness = config.get('enable_sidewall_roughness', False)
        self.num_disorder_runs = config.get('num_disorder_runs', 10)
        
    def generate_disorder_parameters(self, base_params: ExtendedDesignParameters, 
                                   random_seed: Optional[int] = None) -> List[ExtendedDesignParameters]:
        """
        Generate multiple parameter sets with fabrication disorder.
        
        Args:
            base_params: Nominal design parameters
            random_seed: Random seed for reproducibility
            
        Returns:
            List of parameter sets with disorder applied
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        disorder_params = []
        
        for i in range(self.num_disorder_runs):
            # Copy base parameters
            disordered = ExtendedDesignParameters(
                a=base_params.a,
                b=base_params.b, 
                r=base_params.r,
                w=base_params.w,
                N_cells=base_params.N_cells,
                coupling_gap=base_params.coupling_gap,
                coupling_width=base_params.coupling_width
            )
            
            # Apply hole radius disorder (primary error from thesis)
            r_disorder = np.random.normal(0, self.hole_radius_std * base_params.r)
            disordered.r = max(0.01, base_params.r + r_disorder)  # Ensure positive radius
            
            # Apply sidewall roughness (if enabled)
            if self.enable_sidewall_roughness:
                # Model as effective reduction in hole radius
                sidewall_effect = np.random.normal(0, self.sidewall_roughness_std)
                disordered.r = max(0.01, disordered.r + sidewall_effect)
            
            # Recalculate derived parameters
            disordered.R = disordered.calculate_ring_radius()
            
            disorder_params.append(disordered)
            
        return disorder_params


class TopologicalPhotonicCrystalProblem(Problem):
    """Multi-objective optimization problem for topological photonic crystals."""
    
    def __init__(self, config: Dict[str, Any], simulation_function):
        """
        Initialize optimization problem.
        
        Args:
            config: Configuration dictionary
            simulation_function: Function to evaluate designs
        """
        self.config = config
        self.simulation_function = simulation_function
        self.constraints = PhysicsInformedConstraints(
            min_feature_size=config.get('min_feature_size', 0.05)
        )
        self.disorder_model = EnhancedDisorderModel(config.get('disorder', {}))
        
        # Define variable bounds from config
        bounds = config['design_space']
        
        # Define variable bounds for pymoo
        xl = np.array([bounds['a'][0], bounds['b'][0], bounds['r'][0], bounds['w'][0], 
                      bounds['N_cells'][0], bounds['coupling_gap'][0], bounds['coupling_width'][0]])
        xu = np.array([bounds['a'][1], bounds['b'][1], bounds['r'][1], bounds['w'][1],
                      bounds['N_cells'][1], bounds['coupling_gap'][1], bounds['coupling_width'][1]])
        
        # Store variable info for mixed integer handling
        self.var_types = ['real', 'real', 'real', 'real', 'int', 'real', 'real']
        
        # Initialize parent with 4 objectives (q_avg, q_std, bandgap, mode_volume)
        super().__init__(
            n_var=7,
            n_obj=4,
            n_constr=0,  # We'll use penalty methods instead
            xl=xl,
            xu=xu
        )
        
        # Track evaluations for analysis
        self.evaluation_history = []
    
    def _evaluate(self, X, out, *args, **kwargs):
        """Evaluate population of designs."""
        n_pop = X.shape[0]
        objectives = np.zeros((n_pop, self.n_obj))
        
        for i in range(n_pop):
            # Convert to design parameters (ensure N_cells is integer)
            x_vec = X[i].copy()
            x_vec[4] = int(round(x_vec[4]))  # Convert N_cells to integer
            params = ExtendedDesignParameters.from_optimization_vector(x_vec)
            
            # Check constraints and apply penalties
            is_feasible, violations = self.constraints.check_constraints(params)
            
            if is_feasible:
                # Evaluate with disorder
                objectives_obj = self._evaluate_single_design(params)
            else:
                # Apply heavy penalty for infeasible designs
                penalty = self.constraints.constraint_penalty(params)
                objectives_obj = OptimizationObjectives(
                    q_avg=0, q_std=penalty, 
                    bandgap_size=0, mode_volume=penalty
                )
            
            objectives[i] = objectives_obj.to_array()
            
            # Store evaluation history
            self.evaluation_history.append({
                'parameters': params,
                'objectives': objectives_obj,
                'feasible': is_feasible,
                'violations': violations if not is_feasible else []
            })
        
        out["F"] = objectives
    
    def _evaluate_single_design(self, params: ExtendedDesignParameters) -> OptimizationObjectives:
        """Evaluate a single design with disorder analysis."""
        print(f"Evaluating design: R={params.R:.3f}, a={params.a:.3f}, b={params.b:.3f}, "
              f"r={params.r:.3f}, N={params.N_cells}")
        
        # Generate disorder parameter sets
        disorder_params = self.disorder_model.generate_disorder_parameters(params)
        
        # Evaluate each disorder case
        q_factors = []
        bandgaps = []
        mode_volumes = []
        
        for i, disordered_params in enumerate(disorder_params):
            # Convert to legacy format for simulation function
            design_vector = [disordered_params.a, disordered_params.b, 
                           disordered_params.r, disordered_params.R, disordered_params.w]
            
            # Run simulation (this calls either mock or MEEP)
            result = self.simulation_function(design_vector, self.config)
            
            # For now, extract Q-factor from score (legacy format)
            # TODO: Modify simulation function to return full objectives
            if isinstance(result, dict):
                q_factors.append(result.get('q_factor', 0))
                bandgaps.append(result.get('bandgap_size', 0))
                mode_volumes.append(result.get('mode_volume', 1))
            else:
                # Legacy score format - approximate Q-factor
                q_factor = max(0, result + 20000)  # Rough conversion from score
                q_factors.append(q_factor)
                bandgaps.append(params.a - params.b)  # Simple bandgap proxy
                mode_volumes.append(np.pi * params.r**2)  # Simple mode volume proxy
        
        # Calculate statistics
        q_avg = np.mean(q_factors)
        q_std = np.std(q_factors)
        bandgap_avg = np.mean(bandgaps)
        mode_volume_avg = np.mean(mode_volumes)
        
        return OptimizationObjectives(
            q_avg=q_avg,
            q_std=q_std, 
            bandgap_size=bandgap_avg,
            mode_volume=mode_volume_avg
        )


class MultiObjectiveOptimizer:
    """Main multi-objective optimization class."""
    
    def __init__(self, config: Dict[str, Any], simulation_function):
        """Initialize optimizer."""
        self.config = config
        self.simulation_function = simulation_function
        self.problem = TopologicalPhotonicCrystalProblem(config, simulation_function)
        
        # Setup NSGA-III algorithm
        self._setup_algorithm()
        
    def _setup_algorithm(self):
        """Setup NSGA-III algorithm with reference directions."""
        # Generate reference directions for 4 objectives
        ref_dirs = get_reference_directions(
            "das-dennis", 
            n_dim=4, 
            n_partitions=self.config.get('optimizer', {}).get('n_partitions', 3)
        )
        
        # Use NSGA3 with real-coded genetic algorithm (handle integer manually)
        self.algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=self.config.get('optimizer', {}).get('population_size', 50)
        )
    
    def optimize(self, n_generations: int = 100) -> Result:
        """
        Run multi-objective optimization.
        
        Args:
            n_generations: Number of generations to run
            
        Returns:
            Optimization result with Pareto front
        """
        print(f"Starting multi-objective optimization with NSGA-III...")
        print(f"Population size: {self.algorithm.pop_size}")
        print(f"Generations: {n_generations}")
        print(f"Objectives: Q-factor avg, Q-factor std, Bandgap size, Mode volume")
        
        start_time = time.time()
        
        # Run optimization
        result = minimize(
            self.problem,
            self.algorithm,
            termination=('n_gen', n_generations),
            verbose=True
        )
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        return result
    
    def analyze_pareto_front(self, result: Result) -> pd.DataFrame:
        """Analyze and return Pareto front solutions."""
        # Extract Pareto optimal solutions
        pareto_f = result.F
        pareto_x = result.X
        
        pareto_data = []
        for i in range(len(pareto_f)):
            params = ExtendedDesignParameters.from_optimization_vector(pareto_x[i])
            
            # Convert objectives back to positive values
            q_avg = -pareto_f[i][0]
            q_std = pareto_f[i][1] 
            bandgap = -pareto_f[i][2]
            mode_vol = pareto_f[i][3]
            
            pareto_data.append({
                'a': params.a,
                'b': params.b,
                'r': params.r,
                'w': params.w,
                'N_cells': params.N_cells,
                'R': params.R,
                'coupling_gap': params.coupling_gap,
                'coupling_width': params.coupling_width,
                'q_avg': q_avg,
                'q_std': q_std,
                'q_robustness': q_avg / max(q_std, 1e-6),  # Robustness metric
                'bandgap_size': bandgap,
                'mode_volume': mode_vol,
                'dimerization_ratio': params.a / max(params.b, 1e-6)
            })
        
        return pd.DataFrame(pareto_data)
    
    def save_results(self, result: Result, output_dir: str):
        """Save optimization results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Pareto front
        pareto_df = self.analyze_pareto_front(result)
        pareto_df.to_csv(os.path.join(output_dir, 'pareto_front.csv'), index=False)
        
        # Save full evaluation history
        history_df = pd.DataFrame(self.problem.evaluation_history)
        history_df.to_pickle(os.path.join(output_dir, 'evaluation_history.pkl'))
        
        # Save configuration
        with open(os.path.join(output_dir, 'multi_objective_config.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        
        print(f"Results saved to {output_dir}")
        print(f"Pareto front contains {len(pareto_df)} solutions")
        
        return pareto_df


def main():
    """Example usage of multi-objective optimizer."""
    # Example configuration
    config = {
        'design_space': {
            'a': [0.30, 0.60],
            'b': [0.05, 0.20],
            'r': [0.05, 0.18],
            'w': [0.40, 0.70],
            'N_cells': [80, 150],
            'coupling_gap': [0.1, 0.5],
            'coupling_width': [0.3, 0.8]
        },
        'min_feature_size': 0.05,
        'disorder': {
            'hole_radius_disorder_std': 0.05,
            'sidewall_roughness_std': 0.01,
            'enable_sidewall_roughness': False,
            'num_disorder_runs': 5
        },
        'optimizer': {
            'population_size': 20,
            'n_partitions': 3
        }
    }
    
    # Mock simulation function for testing
    def mock_simulation(design_vector, config):
        a, b, r, R, w = design_vector
        # Simple mock that prefers strong dimerization and large rings
        score = (a - b) * 10000 + R * 1000 - r * 5000
        return score + np.random.normal(0, 1000)
    
    # Create optimizer
    optimizer = MultiObjectiveOptimizer(config, mock_simulation)
    
    # Run optimization
    result = optimizer.optimize(n_generations=10)
    
    # Analyze results
    pareto_df = optimizer.analyze_pareto_front(result)
    print("\nPareto Front Summary:")
    print(pareto_df[['a', 'b', 'r', 'R', 'q_avg', 'q_std', 'bandgap_size']].head())


if __name__ == "__main__":
    main()