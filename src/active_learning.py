#!/usr/bin/env python3
"""
Physics-Informed Active Learning for Topological Photonic Crystal Optimization

This module implements advanced active learning strategies that incorporate
physical insights from the AlexisHK thesis to intelligently guide the
multi-objective optimization process.

Key Features:
- Multi-fidelity Gaussian Process surrogate models
- Physics-informed acquisition functions
- Uncertainty quantification with fabrication robustness
- Automated design rule discovery
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional, Callable
from dataclasses import dataclass
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.optimize as opt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal, norm


@dataclass
class SurrogateModelPrediction:
    """Container for surrogate model predictions with uncertainty."""
    mean: np.ndarray
    std: np.ndarray
    confidence_interval: Tuple[np.ndarray, np.ndarray]
    model_confidence: float
    

class PhysicsInformedKernel:
    """
    Custom kernel that incorporates physical insights from topological photonics.
    """
    
    def __init__(self, base_kernel='matern_52', physics_weights=None):
        """
        Initialize physics-informed kernel.
        
        Args:
            base_kernel: Base kernel type ('rbf', 'matern_52', 'matern_32')
            physics_weights: Weights for different physical relationships
        """
        self.base_kernel = base_kernel
        self.physics_weights = physics_weights or {
            'dimerization_strength': 2.0,      # Strong coupling for (a-b) effects
            'ring_radius_scaling': 1.5,        # Ring radius effects on bending loss
            'hole_coupling': 1.0,              # Hole radius effects
            'geometric_coupling': 0.8          # Cross-parameter interactions
        }
        
    def create_kernel(self, length_scales=None):
        """Create the physics-informed kernel."""
        if length_scales is None:
            length_scales = [0.1, 0.1, 0.05, 0.2, 20, 0.1, 0.1]  # Default scales
            
        # Base kernel
        if self.base_kernel == 'rbf':
            base = ConstantKernel(1.0) * RBF(length_scale=length_scales)
        elif self.base_kernel == 'matern_52':
            base = ConstantKernel(1.0) * Matern(length_scale=length_scales, nu=2.5)
        elif self.base_kernel == 'matern_32':
            base = ConstantKernel(1.0) * Matern(length_scale=length_scales, nu=1.5)
        else:
            raise ValueError(f"Unknown kernel type: {self.base_kernel}")
            
        # Add physics-informed structure
        # Shorter length scale for dimerization parameters (a, b) due to strong coupling
        dimerization_kernel = ConstantKernel(self.physics_weights['dimerization_strength']) * \
                            RBF(length_scale=[0.05, 0.05])  # Just for a, b
        
        # Add white noise for robustness
        noise_kernel = WhiteKernel(noise_level=0.01)
        
        return base + noise_kernel


class MultiFidelityGaussianProcess:
    """
    Multi-fidelity Gaussian Process for efficient optimization with different simulation costs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize multi-fidelity GP."""
        self.config = config
        self.physics_kernel = PhysicsInformedKernel()
        
        # High and low fidelity models
        self.high_fidelity_gp = None
        self.low_fidelity_gp = None
        
        # Data storage
        self.high_fidelity_data = {'X': [], 'y': []}
        self.low_fidelity_data = {'X': [], 'y': []}
        
        # Scalers for normalization
        self.x_scaler = StandardScaler()
        self.y_scalers = {}
        
        # Model performance tracking
        self.model_performance = {
            'high_fidelity': {'mse': [], 'r2': []},
            'low_fidelity': {'mse': [], 'r2': []}
        }
        
    def add_training_data(self, X: np.ndarray, y: Dict[str, np.ndarray], fidelity: str):
        """
        Add training data for the specified fidelity level.
        
        Args:
            X: Input parameters [n_samples, n_features]
            y: Dictionary of objectives {objective_name: values}
            fidelity: 'high' or 'low'
        """
        if fidelity == 'high':
            self.high_fidelity_data['X'].append(X)
            if not self.high_fidelity_data['y']:
                self.high_fidelity_data['y'] = {key: [] for key in y.keys()}
            for key, values in y.items():
                self.high_fidelity_data['y'][key].append(values)
        else:
            self.low_fidelity_data['X'].append(X)
            if not self.low_fidelity_data['y']:
                self.low_fidelity_data['y'] = {key: [] for key in y.keys()}
            for key, values in y.items():
                self.low_fidelity_data['y'][key].append(values)
                
    def fit_models(self):
        """Fit both high and low fidelity models."""
        # Fit high fidelity model if data available
        if self.high_fidelity_data['X']:
            X_high = np.vstack(self.high_fidelity_data['X'])
            self.high_fidelity_gp = {}
            
            for objective in self.high_fidelity_data['y'].keys():
                y_high = np.concatenate(self.high_fidelity_data['y'][objective])
                
                # Create physics-informed kernel
                kernel = self.physics_kernel.create_kernel()
                
                # Fit GP
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=3
                )
                
                # Normalize inputs
                if len(X_high) == 1:
                    X_high_norm = self.x_scaler.fit_transform(X_high.reshape(1, -1))
                else:
                    X_high_norm = self.x_scaler.fit_transform(X_high)
                
                gp.fit(X_high_norm, y_high)
                self.high_fidelity_gp[objective] = gp
                
        # Fit low fidelity model
        if self.low_fidelity_data['X']:
            X_low = np.vstack(self.low_fidelity_data['X'])
            self.low_fidelity_gp = {}
            
            for objective in self.low_fidelity_data['y'].keys():
                y_low = np.concatenate(self.low_fidelity_data['y'][objective])
                
                # Simpler kernel for low fidelity
                kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(0.01)
                
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-5,
                    normalize_y=True,
                    n_restarts_optimizer=2
                )
                
                # Use existing scaler or fit new one
                if hasattr(self.x_scaler, 'scale_'):
                    X_low_norm = self.x_scaler.transform(X_low)
                else:
                    X_low_norm = self.x_scaler.fit_transform(X_low)
                
                gp.fit(X_low_norm, y_low)
                self.low_fidelity_gp[objective] = gp
                
    def predict(self, X: np.ndarray, fidelity: str = 'high', 
                return_uncertainty: bool = True) -> Dict[str, SurrogateModelPrediction]:
        """
        Predict objectives using the appropriate fidelity model.
        
        Args:
            X: Input parameters [n_samples, n_features]
            fidelity: 'high' or 'low'
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary of predictions for each objective
        """
        predictions = {}
        
        # Choose appropriate model
        if fidelity == 'high' and self.high_fidelity_gp:
            gp_models = self.high_fidelity_gp
        elif self.low_fidelity_gp:
            gp_models = self.low_fidelity_gp
        else:
            raise ValueError(f"No trained model available for fidelity: {fidelity}")
            
        # Normalize inputs
        X_norm = self.x_scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        
        for objective, gp in gp_models.items():
            if return_uncertainty:
                mean, std = gp.predict(X_norm, return_std=True)
                
                # Calculate confidence intervals (95%)
                ci_lower = mean - 1.96 * std
                ci_upper = mean + 1.96 * std
                
                predictions[objective] = SurrogateModelPrediction(
                    mean=mean,
                    std=std,
                    confidence_interval=(ci_lower, ci_upper),
                    model_confidence=self._calculate_model_confidence(gp, X_norm)
                )
            else:
                mean = gp.predict(X_norm)
                predictions[objective] = SurrogateModelPrediction(
                    mean=mean,
                    std=np.zeros_like(mean),
                    confidence_interval=(mean, mean),
                    model_confidence=1.0
                )
                
        return predictions
    
    def _calculate_model_confidence(self, gp, X_norm):
        """Calculate confidence in model predictions based on training data coverage."""
        # Simple heuristic: confidence decreases with distance from training data
        if hasattr(gp, 'X_train_'):
            distances = np.min(np.linalg.norm(gp.X_train_ - X_norm, axis=1))
            confidence = np.exp(-distances / 0.5)  # Exponential decay
            return np.clip(confidence, 0.1, 1.0)
        return 0.5


class PhysicsInformedAcquisitionFunctions:
    """
    Collection of acquisition functions that incorporate physical insights.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize acquisition functions."""
        self.config = config
        
    def expected_improvement(self, predictions: Dict[str, SurrogateModelPrediction], 
                           current_best: Dict[str, float], 
                           xi: float = 0.01) -> float:
        """
        Multi-objective Expected Improvement.
        """
        total_ei = 0.0
        
        for objective, pred in predictions.items():
            mean = pred.mean[0] if pred.mean.ndim > 0 else pred.mean
            std = pred.std[0] if pred.std.ndim > 0 else pred.std
            
            if std < 1e-6:
                ei = 0.0
            else:
                best_val = current_best.get(objective, 0)
                
                # For minimization objectives (q_std, mode_volume)
                if objective in ['q_std', 'mode_volume']:
                    improvement = best_val - mean - xi
                else:  # For maximization objectives (q_factor, bandgap_size)
                    improvement = mean - best_val - xi
                    
                z = improvement / std
                ei = improvement * norm.cdf(z) + std * norm.pdf(z)
                
            total_ei += ei
            
        return total_ei
    
    def uncertainty_sampling(self, predictions: Dict[str, SurrogateModelPrediction]) -> float:
        """
        Acquisition function that prioritizes high uncertainty regions.
        """
        total_uncertainty = 0.0
        
        for objective, pred in predictions.items():
            std = pred.std[0] if pred.std.ndim > 0 else pred.std
            total_uncertainty += std
            
        return total_uncertainty
    
    def physics_informed_acquisition(self, X: np.ndarray, 
                                   predictions: Dict[str, SurrogateModelPrediction],
                                   physics_weights: Dict[str, float] = None) -> float:
        """
        Custom acquisition function incorporating physical insights from thesis.
        """
        if physics_weights is None:
            physics_weights = {
                'dimerization_preference': 2.0,    # Prefer strong dimerization
                'ring_size_preference': 1.0,       # Prefer larger rings
                'fabrication_feasibility': 3.0,    # Strongly prefer feasible designs
                'uncertainty_exploration': 1.0      # Balance with uncertainty
            }
            
        # Extract design parameters
        a, b, r, R, w, N_cells, coupling_gap = X[:7]
        
        # Physics-based scoring
        physics_score = 0.0
        
        # 1. Dimerization strength (from thesis: key factor)
        dimerization_ratio = a / max(b, 1e-6)
        if 2.0 <= dimerization_ratio <= 6.0:  # Sweet spot from thesis
            physics_score += physics_weights['dimerization_preference'] * \
                           np.exp(-((dimerization_ratio - 3.5)**2) / (2 * 1.0**2))
        
        # 2. Ring size preference (larger is generally better for Q-factor)
        if R > 8.0:  # Above minimum for meaningful bending loss reduction
            physics_score += physics_weights['ring_size_preference'] * (R - 8.0) / 10.0
            
        # 3. Fabrication feasibility (ensure features are manufacturable)
        hole_spacing = b - 2 * r
        edge_clearance = (w - 2 * r) / 2
        min_feature = 0.08  # 80nm minimum
        
        if hole_spacing > min_feature and edge_clearance > min_feature:
            physics_score += physics_weights['fabrication_feasibility']
        elif hole_spacing > 0 and edge_clearance > 0:
            physics_score += physics_weights['fabrication_feasibility'] * 0.5
            
        # 4. Uncertainty exploration
        uncertainty_score = self.uncertainty_sampling(predictions)
        physics_score += physics_weights['uncertainty_exploration'] * uncertainty_score
        
        return physics_score
    
    def hypervolume_improvement(self, predictions: Dict[str, SurrogateModelPrediction],
                              pareto_front: np.ndarray, 
                              reference_point: np.ndarray) -> float:
        """
        Hypervolume improvement for multi-objective optimization.
        """
        # Extract mean predictions
        pred_point = np.array([pred.mean[0] if pred.mean.ndim > 0 else pred.mean 
                              for pred in predictions.values()])
        
        # Simple hypervolume approximation
        # In practice, would use proper hypervolume calculation
        if len(pareto_front) == 0:
            return np.prod(np.maximum(0, reference_point - pred_point))
        
        # Check if point dominates any existing points
        dominates = np.all(pred_point <= pareto_front, axis=1)
        if np.any(dominates):
            return 1.0  # High score for dominating points
        
        # Check if point is dominated
        dominated = np.any(np.all(pareto_front <= pred_point, axis=1))
        if dominated:
            return 0.0  # Low score for dominated points
            
        return 0.5  # Medium score for non-dominated points


class ActiveLearningOptimizer:
    """
    Main active learning optimizer that coordinates surrogate models and acquisition functions.
    """
    
    def __init__(self, config: Dict[str, Any], simulation_function: Callable):
        """Initialize active learning optimizer."""
        self.config = config
        self.simulation_function = simulation_function
        
        # Initialize components
        self.surrogate_model = MultiFidelityGaussianProcess(config)
        self.acquisition_functions = PhysicsInformedAcquisitionFunctions(config)
        
        # Optimization history
        self.evaluation_history = []
        self.current_pareto_front = np.array([])
        self.current_best = {}
        
        # Active learning parameters
        self.n_initial_samples = config.get('active_learning', {}).get('n_initial_samples', 20)
        self.n_iterations = config.get('active_learning', {}).get('n_iterations', 50)
        self.fidelity_schedule = self._create_fidelity_schedule()
        
    def _create_fidelity_schedule(self):
        """Create schedule for fidelity allocation during optimization."""
        # Start with more low-fidelity, gradually increase high-fidelity
        schedule = []
        for i in range(self.n_iterations):
            if i < self.n_iterations * 0.3:
                high_fidelity_prob = 0.1  # 10% high-fidelity early on
            elif i < self.n_iterations * 0.7:
                high_fidelity_prob = 0.3  # 30% in middle phase
            else:
                high_fidelity_prob = 0.6  # 60% in final phase
                
            schedule.append(high_fidelity_prob)
            
        return schedule
    
    def optimize(self):
        """Run active learning optimization."""
        print(f"Starting physics-informed active learning optimization...")
        print(f"Initial samples: {self.n_initial_samples}")
        print(f"Total iterations: {self.n_iterations}")
        
        # Phase 1: Initial sampling
        self._initial_sampling()
        
        # Phase 2: Active learning iterations
        for iteration in range(self.n_iterations):
            print(f"\nActive Learning Iteration {iteration + 1}/{self.n_iterations}")
            
            # Fit surrogate models
            self.surrogate_model.fit_models()
            
            # Select next sample point
            next_x = self._select_next_sample()
            
            # Determine fidelity
            use_high_fidelity = np.random.random() < self.fidelity_schedule[iteration]
            fidelity = 'high' if use_high_fidelity else 'low'
            
            # Evaluate new point
            objectives = self._evaluate_point(next_x, fidelity)
            
            # Update surrogate models
            self.surrogate_model.add_training_data(
                next_x.reshape(1, -1), objectives, fidelity
            )
            
            # Update Pareto front
            self._update_pareto_front(objectives)
            
        print(f"\nActive learning optimization completed!")
        return self._get_results()
    
    def _initial_sampling(self):
        """Generate initial sample points using physics-informed sampling."""
        print("Generating initial samples with physics-informed strategy...")
        
        # Get parameter bounds
        bounds = self.config['design_space']
        n_params = len(bounds)
        
        # Generate samples
        samples = []
        for i in range(self.n_initial_samples):
            if i < self.n_initial_samples * 0.5:
                # Physics-informed sampling (50%)
                sample = self._physics_informed_sample(bounds)
            else:
                # Random sampling (50%)
                sample = self._random_sample(bounds)
                
            samples.append(sample)
            
        # Evaluate initial samples
        for i, sample in enumerate(samples):
            print(f"  Initial sample {i+1}/{self.n_initial_samples}")
            fidelity = 'high' if i % 3 == 0 else 'low'  # 1/3 high-fidelity
            objectives = self._evaluate_point(sample, fidelity)
            
            self.surrogate_model.add_training_data(
                sample.reshape(1, -1), objectives, fidelity
            )
            
    def _physics_informed_sample(self, bounds):
        """Generate a physics-informed sample point."""
        sample = np.zeros(7)
        
        # Sample dimerization parameters with preference for strong dimerization
        a_range = bounds['a']
        b_range = bounds['b']
        
        # Prefer higher a and lower b for strong dimerization
        sample[0] = np.random.beta(2, 1) * (a_range[1] - a_range[0]) + a_range[0]  # a
        sample[1] = np.random.beta(1, 2) * (b_range[1] - b_range[0]) + b_range[0]  # b
        
        # Hole radius with preference for fabricability
        r_range = bounds['r']
        sample[2] = np.random.normal(0.12, 0.02)  # Prefer r ≈ 0.12
        sample[2] = np.clip(sample[2], r_range[0], r_range[1])
        
        # Waveguide width
        w_range = bounds['w']
        sample[3] = np.random.uniform(w_range[0], w_range[1])
        
        # Number of cells (discrete)
        N_range = bounds['N_cells']
        sample[4] = np.random.randint(N_range[0], N_range[1] + 1)
        
        # Coupling parameters
        sample[5] = np.random.uniform(bounds['coupling_gap'][0], bounds['coupling_gap'][1])
        sample[6] = np.random.uniform(bounds['coupling_width'][0], bounds['coupling_width'][1])
        
        return sample
    
    def _random_sample(self, bounds):
        """Generate a random sample point."""
        sample = np.zeros(7)
        param_names = ['a', 'b', 'r', 'w', 'N_cells', 'coupling_gap', 'coupling_width']
        
        for i, param in enumerate(param_names):
            param_range = bounds[param]
            if param == 'N_cells':
                sample[i] = np.random.randint(param_range[0], param_range[1] + 1)
            else:
                sample[i] = np.random.uniform(param_range[0], param_range[1])
                
        return sample
    
    def _select_next_sample(self):
        """Select next sample point using acquisition function."""
        bounds = self.config['design_space']
        
        # Define optimization bounds for scipy
        opt_bounds = []
        for param in ['a', 'b', 'r', 'w', 'N_cells', 'coupling_gap', 'coupling_width']:
            opt_bounds.append(bounds[param])
            
        # Acquisition function to minimize (negative because scipy minimizes)
        def neg_acquisition(x):
            # Predict with surrogate model
            try:
                predictions = self.surrogate_model.predict(x, fidelity='high')
                
                # Combine different acquisition strategies
                ei_score = self.acquisition_functions.expected_improvement(
                    predictions, self.current_best
                )
                
                uncertainty_score = self.acquisition_functions.uncertainty_sampling(predictions)
                
                physics_score = self.acquisition_functions.physics_informed_acquisition(
                    x, predictions
                )
                
                # Weighted combination
                total_score = (0.4 * ei_score + 0.3 * uncertainty_score + 0.3 * physics_score)
                
                return -total_score  # Negative for minimization
                
            except Exception as e:
                print(f"Error in acquisition function: {e}")
                return 1e6  # Large penalty for errors
        
        # Multi-start optimization
        best_x = None
        best_score = np.inf
        
        for _ in range(10):  # 10 random starts
            x0 = self._random_sample(bounds)
            
            try:
                result = opt.minimize(
                    neg_acquisition, x0, bounds=opt_bounds, method='L-BFGS-B'
                )
                
                if result.success and result.fun < best_score:
                    best_score = result.fun
                    best_x = result.x
                    
            except Exception as e:
                print(f"Optimization error: {e}")
                continue
                
        if best_x is None:
            # Fallback to random sample
            best_x = self._random_sample(bounds)
            
        # Ensure N_cells is integer
        best_x[4] = int(round(best_x[4]))
        
        return best_x
    
    def _evaluate_point(self, x, fidelity):
        """Evaluate a design point with specified fidelity."""
        # Convert to legacy format for simulation function
        design_vector = x[:5]  # [a, b, r, R, w] - but R needs to be calculated
        
        # Calculate R from constraint: 2πR = N * (a + b)
        a, b, r, w, N_cells = x[0], x[1], x[2], x[3], int(x[4])
        R = (N_cells * (a + b)) / (2 * np.pi)
        design_vector[3] = R
        
        # Set up config for specified fidelity
        eval_config = self.config.copy()
        if fidelity == 'high':
            eval_config.update(eval_config.get('simulation', {}).get('high_fidelity', {}))
        else:
            eval_config.update(eval_config.get('simulation', {}).get('low_fidelity', {}))
            
        eval_config['return_comprehensive_objectives'] = True
        
        # Evaluate
        result = self.simulation_function(design_vector, eval_config)
        
        # Store evaluation history
        self.evaluation_history.append({
            'x': x.copy(),
            'design_vector': design_vector.copy(),
            'fidelity': fidelity,
            'objectives': result.copy() if isinstance(result, dict) else {'score': result}
        })
        
        return result if isinstance(result, dict) else {'score': result}
    
    def _update_pareto_front(self, objectives):
        """Update current Pareto front and best values."""
        # Update best values for each objective
        for key, value in objectives.items():
            if key not in self.current_best:
                self.current_best[key] = value
            else:
                if key in ['q_std', 'mode_volume']:  # Minimization objectives
                    if value < self.current_best[key]:
                        self.current_best[key] = value
                else:  # Maximization objectives
                    if value > self.current_best[key]:
                        self.current_best[key] = value
    
    def _get_results(self):
        """Get optimization results."""
        results_df = pd.DataFrame(self.evaluation_history)
        
        return {
            'evaluation_history': results_df,
            'current_best': self.current_best,
            'surrogate_model': self.surrogate_model,
            'total_evaluations': len(self.evaluation_history)
        }


def main():
    """Example usage of active learning optimizer."""
    # Example configuration
    config = {
        'design_space': {
            'a': [0.30, 0.60],
            'b': [0.05, 0.20],
            'r': [0.08, 0.16],
            'w': [0.45, 0.65],
            'N_cells': [100, 200],
            'coupling_gap': [0.15, 0.35],
            'coupling_width': [0.45, 0.65]
        },
        'active_learning': {
            'n_initial_samples': 10,
            'n_iterations': 20
        },
        'simulation': {
            'high_fidelity': {'resolution': 35},
            'low_fidelity': {'resolution': 20}
        }
    }
    
    # Mock simulation function
    def mock_simulation(design_vector, config):
        a, b, r, R, w = design_vector
        
        # Simple physics model
        q_factor = 15000 + (a - b) * 20000 + R * 1000 - (r - 0.12)**2 * 50000
        q_std = 1000 + np.random.normal(0, 500)
        bandgap = (a - b) * 50
        mode_volume = np.pi * r**2 * 0.5
        
        return {
            'q_factor': q_factor + np.random.normal(0, 1000),
            'q_std': max(500, q_std),
            'bandgap_size': max(1, bandgap + np.random.normal(0, 2)),
            'mode_volume': max(0.1, mode_volume + np.random.normal(0, 0.05))
        }
    
    # Run optimization
    optimizer = ActiveLearningOptimizer(config, mock_simulation)
    results = optimizer.optimize()
    
    print(f"\nOptimization completed!")
    print(f"Total evaluations: {results['total_evaluations']}")
    print(f"Current best objectives: {results['current_best']}")


if __name__ == "__main__":
    main()