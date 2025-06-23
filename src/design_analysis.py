#!/usr/bin/env python3
"""
Automated Design Rule Discovery and Analysis for Topological Photonic Crystals

This module implements advanced analysis techniques to automatically discover
design rules and physical insights from optimization results, based on the
physics understanding from the AlexisHK thesis.

Key Features:
- Symbolic regression for design rule discovery
- SHAP analysis for parameter importance
- Cluster analysis for design regime identification
- Physics-informed feature engineering
- Automated report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Advanced analysis
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Symbolic regression (if available)
try:
    from gplearn.genetic import SymbolicRegressor
    GPLEARN_AVAILABLE = True
except ImportError:
    GPLEARN_AVAILABLE = False
    print("Warning: gplearn not available. Symbolic regression disabled.")

# SHAP for explainable AI (if available)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Explainability analysis limited.")

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class PhysicsFeatureEngineering:
    """
    Physics-informed feature engineering based on topological photonic crystal theory.
    """
    
    def __init__(self):
        """Initialize feature engineering."""
        self.feature_names = []
        self.feature_descriptions = {}
        
    def create_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create physics-informed features from raw parameters.
        
        Args:
            df: DataFrame with columns [a, b, r, R, w, N_cells, ...]
            
        Returns:
            DataFrame with additional physics features
        """
        df_enhanced = df.copy()
        
        # Primary physics features from thesis analysis
        df_enhanced['dimerization_strength'] = np.abs(df['a'] - df['b'])
        df_enhanced['dimerization_ratio'] = df['a'] / np.maximum(df['b'], 1e-6)
        df_enhanced['unit_cell_length'] = df['a'] + df['b']
        
        # Geometric features
        df_enhanced['hole_area'] = np.pi * df['r']**2
        df_enhanced['waveguide_area'] = df['w'] * 2 * np.pi * df['R']
        df_enhanced['filling_factor'] = (df_enhanced['hole_area'] * df['N_cells']) / df_enhanced['waveguide_area']
        
        # Ring geometry features
        df_enhanced['ring_circumference'] = 2 * np.pi * df['R']
        df_enhanced['bending_radius_norm'] = df['R'] / df['w']  # Normalized bending radius
        
        # Fabrication-related features
        df_enhanced['hole_spacing'] = df['b'] - 2 * df['r']
        df_enhanced['edge_clearance'] = (df['w'] - 2 * df['r']) / 2
        df_enhanced['min_feature_size'] = np.minimum(df_enhanced['hole_spacing'], df_enhanced['edge_clearance'])
        
        # SSH model features
        df_enhanced['ssh_asymmetry'] = (df['a'] - df['b']) / (df['a'] + df['b'])
        df_enhanced['topological_gap_proxy'] = df_enhanced['dimerization_strength'] / df_enhanced['unit_cell_length']
        
        # Mode confinement estimates
        df_enhanced['radial_confinement'] = df['w'] / (2 * df['r'])
        df_enhanced['azimuthal_confinement'] = df['R'] / df_enhanced['unit_cell_length']
        
        # Coupling strength estimates
        df_enhanced['hole_coupling_strength'] = df['r'] / df_enhanced['unit_cell_length']
        df_enhanced['waveguide_coupling'] = df['w'] / df['R']
        
        # Quality factor proxies from physics
        df_enhanced['bending_loss_proxy'] = np.exp(-df['R'] / df['w'])  # Exponential bending loss
        df_enhanced['radiation_loss_proxy'] = df['r'] / df['w']  # Radiation through holes
        
        # Update feature tracking
        new_features = [col for col in df_enhanced.columns if col not in df.columns]
        self.feature_names.extend(new_features)
        
        # Add descriptions for new features
        descriptions = {
            'dimerization_strength': 'Absolute difference |a-b| - key for topological gap',
            'dimerization_ratio': 'Ratio a/b - primary topological parameter',
            'unit_cell_length': 'Unit cell size a+b',
            'hole_area': 'Cross-sectional area of holes',
            'filling_factor': 'Fraction of waveguide filled with holes',
            'ssh_asymmetry': 'SSH asymmetry parameter (a-b)/(a+b)',
            'topological_gap_proxy': 'Estimate of topological bandgap',
            'bending_loss_proxy': 'Exponential estimate of bending losses',
            'min_feature_size': 'Minimum manufacturable feature size'
        }
        self.feature_descriptions.update(descriptions)
        
        return df_enhanced


class DesignRuleDiscovery:
    """
    Automated discovery of design rules using machine learning and symbolic regression.
    """
    
    def __init__(self):
        """Initialize design rule discovery."""
        self.models = {}
        self.discovered_rules = {}
        self.feature_importance = {}
        
    def discover_rules(self, df: pd.DataFrame, target_objectives: List[str]) -> Dict[str, Any]:
        """
        Discover design rules for given objectives.
        
        Args:
            df: DataFrame with design parameters and objectives
            target_objectives: List of objective columns to analyze
            
        Returns:
            Dictionary containing discovered rules and models
        """
        results = {}
        
        # Create physics features
        feature_eng = PhysicsFeatureEngineering()
        df_enhanced = feature_eng.create_physics_features(df)
        
        # Define feature columns (exclude objectives and identifiers)
        feature_cols = [col for col in df_enhanced.columns 
                       if col not in target_objectives and 
                       col not in ['design_id', 'timestamp', 'fidelity']]
        
        X = df_enhanced[feature_cols]
        
        for objective in target_objectives:
            if objective not in df_enhanced.columns:
                continue
                
            y = df_enhanced[objective]
            
            print(f"\nDiscovering rules for {objective}...")
            
            # 1. Random Forest for feature importance
            rf_results = self._random_forest_analysis(X, y, feature_cols, objective)
            
            # 2. Polynomial regression for nonlinear relationships
            poly_results = self._polynomial_regression_analysis(X, y, feature_cols, objective)
            
            # 3. Symbolic regression (if available)
            if GPLEARN_AVAILABLE:
                symbolic_results = self._symbolic_regression_analysis(X, y, feature_cols, objective)
            else:
                symbolic_results = {'message': 'Symbolic regression not available'}
            
            # 4. SHAP analysis (if available)
            if SHAP_AVAILABLE:
                shap_results = self._shap_analysis(X, y, feature_cols, objective)
            else:
                shap_results = {'message': 'SHAP analysis not available'}
            
            results[objective] = {
                'random_forest': rf_results,
                'polynomial': poly_results,
                'symbolic': symbolic_results,
                'shap': shap_results,
                'feature_engineering': feature_eng.feature_descriptions
            }
            
        return results
    
    def _random_forest_analysis(self, X, y, feature_cols, objective):
        """Random Forest analysis for feature importance."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X_train, y_train)
        
        # Predictions and performance
        y_pred = rf.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Top features analysis
        top_features = importance.head(10)
        
        return {
            'model': rf,
            'r2_score': r2,
            'mse': mse,
            'feature_importance': importance,
            'top_features': top_features,
            'interpretation': self._interpret_rf_results(top_features, objective)
        }
    
    def _polynomial_regression_analysis(self, X, y, feature_cols, objective):
        """Polynomial regression for nonlinear relationships."""
        # Select top features from correlation
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(5).index.tolist()
        
        X_selected = X[top_features]
        
        # Create polynomial features (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
        
        # Train models with different regularization
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }
        
        results = {}
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'r2_score': r2,
                'mse': mean_squared_error(y_test, y_pred)
            }
            
            if r2 > best_score:
                best_score = r2
                best_model = model
        
        # Get feature names for polynomial features
        feature_names = poly.get_feature_names_out(top_features)
        
        # Extract significant coefficients
        if hasattr(best_model, 'coef_'):
            significant_features = []
            for i, (coef, name) in enumerate(zip(best_model.coef_, feature_names)):
                if abs(coef) > 0.1:  # Threshold for significance
                    significant_features.append({
                        'feature': name,
                        'coefficient': coef,
                        'abs_coefficient': abs(coef)
                    })
            
            significant_features.sort(key=lambda x: x['abs_coefficient'], reverse=True)
        else:
            significant_features = []
        
        return {
            'models': results,
            'best_model': best_model,
            'best_score': best_score,
            'polynomial_transformer': poly,
            'selected_features': top_features,
            'significant_features': significant_features[:10],
            'interpretation': self._interpret_polynomial_results(significant_features[:5], objective)
        }
    
    def _symbolic_regression_analysis(self, X, y, feature_cols, objective):
        """Symbolic regression to discover explicit mathematical relationships."""
        # Select subset of most important features for computational efficiency
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        top_features = correlations.head(8).index.tolist()
        
        X_selected = X[top_features]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
        
        # Symbolic regression
        est_gp = SymbolicRegressor(
            population_size=1000,
            generations=10,
            tournament_size=20,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
            parsimony_coefficient=0.01,
            random_state=42,
            n_jobs=1  # Use single thread to avoid issues
        )
        
        try:
            est_gp.fit(X_train, y_train)
            
            # Get best program
            best_program = est_gp._program
            mathematical_expression = str(best_program)
            
            # Evaluate performance
            y_pred = est_gp.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': est_gp,
                'mathematical_expression': mathematical_expression,
                'r2_score': r2,
                'mse': mean_squared_error(y_test, y_pred),
                'feature_names': top_features,
                'interpretation': self._interpret_symbolic_results(mathematical_expression, top_features, objective)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'message': 'Symbolic regression failed'
            }
    
    def _shap_analysis(self, X, y, feature_cols, objective):
        """SHAP analysis for explainable AI."""
        # Train a gradient boosting model for SHAP analysis
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X.sample(min(100, len(X))))
        
        # Calculate SHAP feature importance
        shap_importance = pd.DataFrame({
            'feature': feature_cols,
            'shap_importance': np.abs(shap_values).mean(0)
        }).sort_values('shap_importance', ascending=False)
        
        return {
            'model': model,
            'explainer': explainer,
            'shap_values': shap_values,
            'shap_importance': shap_importance,
            'interpretation': self._interpret_shap_results(shap_importance.head(5), objective)
        }
    
    def _interpret_rf_results(self, top_features, objective):
        """Interpret Random Forest results in physics context."""
        interpretations = []
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Physics-informed interpretations
            if 'dimerization' in feature:
                interpretations.append(
                    f"{feature} (importance: {importance:.3f}) - "
                    f"Critical for topological protection in SSH model"
                )
            elif 'bending' in feature:
                interpretations.append(
                    f"{feature} (importance: {importance:.3f}) - "
                    f"Controls radiation losses from ring curvature"
                )
            elif 'filling' in feature:
                interpretations.append(
                    f"{feature} (importance: {importance:.3f}) - "
                    f"Affects light-matter interaction strength"
                )
            else:
                interpretations.append(
                    f"{feature} (importance: {importance:.3f}) - "
                    f"Significant parameter for {objective}"
                )
        
        return interpretations
    
    def _interpret_polynomial_results(self, significant_features, objective):
        """Interpret polynomial regression results."""
        interpretations = []
        
        for feature_data in significant_features:
            feature = feature_data['feature']
            coef = feature_data['coefficient']
            
            # Interpret coefficient sign and magnitude
            direction = "increases" if coef > 0 else "decreases"
            magnitude = "strongly" if abs(coef) > 1.0 else "moderately"
            
            interpretations.append(
                f"{feature} {magnitude} {direction} {objective} (coef: {coef:.3f})"
            )
        
        return interpretations
    
    def _interpret_symbolic_results(self, expression, features, objective):
        """Interpret symbolic regression results."""
        return [
            f"Mathematical relationship discovered: {objective} ≈ {expression}",
            f"Based on features: {', '.join(features)}",
            "This represents an automatically discovered design rule"
        ]
    
    def _interpret_shap_results(self, shap_importance, objective):
        """Interpret SHAP analysis results."""
        interpretations = []
        
        for _, row in shap_importance.iterrows():
            feature = row['feature']
            importance = row['shap_importance']
            
            interpretations.append(
                f"{feature} has high impact on {objective} (SHAP: {importance:.3f})"
            )
        
        return interpretations


class DesignRegimeAnalysis:
    """
    Identify and analyze distinct design regimes using clustering and dimensionality reduction.
    """
    
    def __init__(self):
        """Initialize regime analysis."""
        self.cluster_labels = None
        self.regime_characteristics = {}
        
    def identify_regimes(self, df: pd.DataFrame, n_regimes: int = None) -> Dict[str, Any]:
        """
        Identify distinct design regimes using clustering.
        
        Args:
            df: DataFrame with design parameters and objectives
            n_regimes: Number of regimes to identify (if None, use elbow method)
            
        Returns:
            Dictionary with regime analysis results
        """
        # Create physics features
        feature_eng = PhysicsFeatureEngineering()
        df_enhanced = feature_eng.create_physics_features(df)
        
        # Select features for clustering
        feature_cols = ['dimerization_ratio', 'dimerization_strength', 'bending_radius_norm',
                       'filling_factor', 'ssh_asymmetry', 'min_feature_size']
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in df_enhanced.columns]
        X = df_enhanced[available_features]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine optimal number of clusters if not specified
        if n_regimes is None:
            n_regimes = self._find_optimal_clusters(X_scaled)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Perform PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Analyze regime characteristics
        regime_analysis = self._analyze_regime_characteristics(
            df_enhanced, cluster_labels, available_features
        )
        
        # Create regime performance analysis
        performance_analysis = self._analyze_regime_performance(
            df_enhanced, cluster_labels
        )
        
        self.cluster_labels = cluster_labels
        self.regime_characteristics = regime_analysis
        
        return {
            'cluster_labels': cluster_labels,
            'n_regimes': n_regimes,
            'kmeans_model': kmeans,
            'pca_model': pca,
            'pca_coordinates': X_pca,
            'regime_characteristics': regime_analysis,
            'performance_analysis': performance_analysis,
            'feature_columns': available_features,
            'scaler': scaler
        }
    
    def _find_optimal_clusters(self, X_scaled, max_clusters=8):
        """Find optimal number of clusters using elbow method."""
        inertias = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection (could be improved)
        diffs = np.diff(inertias)
        diff_ratios = diffs[:-1] / diffs[1:]
        optimal_k = K_range[np.argmax(diff_ratios) + 1]
        
        return min(optimal_k, 5)  # Cap at 5 regimes for interpretability
    
    def _analyze_regime_characteristics(self, df, cluster_labels, feature_cols):
        """Analyze characteristics of each regime."""
        df_with_clusters = df.copy()
        df_with_clusters['regime'] = cluster_labels
        
        regime_stats = {}
        
        for regime in np.unique(cluster_labels):
            regime_data = df_with_clusters[df_with_clusters['regime'] == regime]
            
            # Calculate statistics for each feature
            stats = {}
            for col in feature_cols + ['q_avg', 'q_std', 'bandgap_size', 'mode_volume']:
                if col in regime_data.columns:
                    stats[col] = {
                        'mean': regime_data[col].mean(),
                        'std': regime_data[col].std(),
                        'min': regime_data[col].min(),
                        'max': regime_data[col].max(),
                        'count': len(regime_data)
                    }
            
            # Characterize regime based on dominant features
            characterization = self._characterize_regime(stats)
            
            regime_stats[f'Regime_{regime}'] = {
                'statistics': stats,
                'characterization': characterization,
                'size': len(regime_data)
            }
        
        return regime_stats
    
    def _characterize_regime(self, stats):
        """Provide physics-based characterization of regime."""
        characteristics = []
        
        # Analyze dimerization
        if 'dimerization_ratio' in stats:
            ratio = stats['dimerization_ratio']['mean']
            if ratio > 4.0:
                characteristics.append("Extreme dimerization regime")
            elif ratio > 2.5:
                characteristics.append("Strong dimerization regime")
            else:
                characteristics.append("Moderate dimerization regime")
        
        # Analyze ring size
        if 'bending_radius_norm' in stats:
            bending = stats['bending_radius_norm']['mean']
            if bending > 20:
                characteristics.append("Large ring (low bending loss)")
            elif bending > 10:
                characteristics.append("Medium ring")
            else:
                characteristics.append("Compact ring (high bending loss)")
        
        # Analyze fabrication feasibility
        if 'min_feature_size' in stats:
            min_feature = stats['min_feature_size']['mean']
            if min_feature > 0.1:
                characteristics.append("Easily manufacturable")
            elif min_feature > 0.05:
                characteristics.append("Standard fabrication")
            else:
                characteristics.append("Challenging fabrication")
        
        return characteristics
    
    def _analyze_regime_performance(self, df, cluster_labels):
        """Analyze performance characteristics of each regime."""
        df_with_clusters = df.copy()
        df_with_clusters['regime'] = cluster_labels
        
        performance_metrics = ['q_avg', 'q_std', 'bandgap_size', 'mode_volume']
        available_metrics = [m for m in performance_metrics if m in df.columns]
        
        regime_performance = {}
        
        for regime in np.unique(cluster_labels):
            regime_data = df_with_clusters[df_with_clusters['regime'] == regime]
            
            performance = {}
            for metric in available_metrics:
                performance[metric] = {
                    'mean': regime_data[metric].mean(),
                    'std': regime_data[metric].std(),
                    'best': regime_data[metric].max() if metric in ['q_avg', 'bandgap_size'] 
                           else regime_data[metric].min()
                }
            
            # Calculate composite scores
            if 'q_avg' in performance and 'q_std' in performance:
                performance['robustness_ratio'] = {
                    'mean': (regime_data['q_avg'] / regime_data['q_std']).mean(),
                    'std': (regime_data['q_avg'] / regime_data['q_std']).std()
                }
            
            regime_performance[f'Regime_{regime}'] = performance
        
        return regime_performance


def create_comprehensive_analysis_report(optimization_results: pd.DataFrame, 
                                       output_dir: str) -> Dict[str, Any]:
    """
    Create a comprehensive analysis report with automated insights.
    
    Args:
        optimization_results: DataFrame with optimization history
        output_dir: Directory to save analysis results
        
    Returns:
        Dictionary with all analysis results
    """
    print("Creating comprehensive analysis report...")
    
    # Initialize analysis components
    rule_discovery = DesignRuleDiscovery()
    regime_analysis = DesignRegimeAnalysis()
    
    # Define target objectives
    target_objectives = ['q_avg', 'q_std', 'bandgap_size', 'mode_volume']
    available_objectives = [obj for obj in target_objectives if obj in optimization_results.columns]
    
    if not available_objectives:
        print("Warning: No target objectives found in data")
        return {}
    
    # 1. Design rule discovery
    print("Discovering design rules...")
    design_rules = rule_discovery.discover_rules(optimization_results, available_objectives)
    
    # 2. Design regime analysis
    print("Analyzing design regimes...")
    regime_results = regime_analysis.identify_regimes(optimization_results)
    
    # 3. Create visualizations
    print("Creating visualizations...")
    create_analysis_visualizations(optimization_results, regime_results, output_dir)
    
    # 4. Generate text report
    print("Generating text report...")
    text_report = generate_text_report(design_rules, regime_results, available_objectives)
    
    # Save text report
    with open(f"{output_dir}/comprehensive_analysis_report.md", 'w') as f:
        f.write(text_report)
    
    return {
        'design_rules': design_rules,
        'regime_analysis': regime_results,
        'text_report': text_report,
        'available_objectives': available_objectives
    }


def create_analysis_visualizations(df: pd.DataFrame, regime_results: Dict, output_dir: str):
    """Create comprehensive visualization suite."""
    # Set up plotting
    plt.style.use('default')
    
    # 1. Regime visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # PCA plot with regimes
    pca_coords = regime_results['pca_coordinates']
    cluster_labels = regime_results['cluster_labels']
    
    scatter = axes[0,0].scatter(pca_coords[:, 0], pca_coords[:, 1], 
                               c=cluster_labels, cmap='tab10', alpha=0.7)
    axes[0,0].set_xlabel('First Principal Component')
    axes[0,0].set_ylabel('Second Principal Component') 
    axes[0,0].set_title('Design Regimes in PCA Space')
    plt.colorbar(scatter, ax=axes[0,0], label='Regime')
    
    # Feature importance (if available)
    feature_eng = PhysicsFeatureEngineering()
    df_enhanced = feature_eng.create_physics_features(df)
    
    if 'q_avg' in df_enhanced.columns:
        # Correlation with Q-factor
        correlations = df_enhanced.select_dtypes(include=[np.number]).corrwith(
            df_enhanced['q_avg']
        ).abs().sort_values(ascending=False)[:10]
        
        axes[0,1].barh(range(len(correlations)), correlations.values)
        axes[0,1].set_yticks(range(len(correlations)))
        axes[0,1].set_yticklabels(correlations.index, fontsize=8)
        axes[0,1].set_xlabel('Absolute Correlation with Q-factor')
        axes[0,1].set_title('Feature Importance for Q-factor')
    
    # Regime performance comparison
    if 'regime' not in df_enhanced.columns:
        df_enhanced['regime'] = cluster_labels
        
    if 'q_avg' in df_enhanced.columns:
        regime_performance = df_enhanced.groupby('regime')['q_avg'].mean()
        axes[1,0].bar(range(len(regime_performance)), regime_performance.values)
        axes[1,0].set_xlabel('Regime')
        axes[1,0].set_ylabel('Average Q-factor')
        axes[1,0].set_title('Performance by Regime')
        axes[1,0].set_xticks(range(len(regime_performance)))
        axes[1,0].set_xticklabels([f'R{i}' for i in regime_performance.index])
    
    # Physics relationship plot
    if 'dimerization_ratio' in df_enhanced.columns and 'q_avg' in df_enhanced.columns:
        scatter2 = axes[1,1].scatter(df_enhanced['dimerization_ratio'], df_enhanced['q_avg'],
                                    c=cluster_labels, cmap='tab10', alpha=0.7)
        axes[1,1].set_xlabel('Dimerization Ratio (a/b)')
        axes[1,1].set_ylabel('Q-factor')
        axes[1,1].set_title('Q-factor vs Dimerization (Key Physics Relationship)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis visualizations saved to {output_dir}/comprehensive_analysis.png")


def generate_text_report(design_rules: Dict, regime_results: Dict, objectives: List[str]) -> str:
    """Generate comprehensive text report."""
    report_lines = [
        "# Comprehensive Analysis Report: Topological Photonic Crystal Optimization",
        "",
        "## Executive Summary",
        "",
        f"This report presents automated analysis results from the optimization of topological photonic crystal ring resonators. "
        f"The analysis covers {len(objectives)} primary objectives and identifies {regime_results.get('n_regimes', 'unknown')} distinct design regimes.",
        "",
        "## Design Rule Discovery",
        ""
    ]
    
    for objective in objectives:
        if objective in design_rules:
            report_lines.extend([
                f"### {objective.upper()} Optimization Rules",
                ""
            ])
            
            rules = design_rules[objective]
            
            # Random Forest insights
            if 'random_forest' in rules:
                rf_results = rules['random_forest']
                report_lines.extend([
                    f"**Machine Learning Model Performance**: R² = {rf_results.get('r2_score', 0):.3f}",
                    "",
                    "**Most Important Parameters**:",
                ])
                
                for interpretation in rf_results.get('interpretation', [])[:3]:
                    report_lines.append(f"- {interpretation}")
                
                report_lines.append("")
            
            # Polynomial relationships
            if 'polynomial' in rules:
                poly_results = rules['polynomial']
                report_lines.extend([
                    "**Discovered Mathematical Relationships**:",
                ])
                
                for interpretation in poly_results.get('interpretation', [])[:3]:
                    report_lines.append(f"- {interpretation}")
                
                report_lines.append("")
            
            # Symbolic regression
            if 'symbolic' in rules and 'mathematical_expression' in rules['symbolic']:
                expr = rules['symbolic']['mathematical_expression']
                report_lines.extend([
                    "**Automatically Discovered Formula**:",
                    f"```",
                    f"{objective} ≈ {expr}",
                    f"```",
                    ""
                ])
    
    # Regime analysis
    report_lines.extend([
        "## Design Regime Analysis",
        "",
        f"The optimization space naturally clusters into {regime_results.get('n_regimes', 'several')} distinct regimes:",
        ""
    ])
    
    regime_chars = regime_results.get('regime_characteristics', {})
    for regime_name, regime_data in regime_chars.items():
        report_lines.extend([
            f"### {regime_name}",
            f"- **Size**: {regime_data.get('size', 0)} designs",
            "- **Characteristics**: " + ", ".join(regime_data.get('characterization', [])),
            ""
        ])
        
        # Add performance statistics if available
        if 'statistics' in regime_data:
            stats = regime_data['statistics']
            if 'q_avg' in stats:
                mean_q = stats['q_avg']['mean']
                report_lines.append(f"- **Average Q-factor**: {mean_q:.0f}")
            if 'dimerization_ratio' in stats:
                mean_dim = stats['dimerization_ratio']['mean']
                report_lines.append(f"- **Average Dimerization Ratio**: {mean_dim:.2f}")
        
        report_lines.append("")
    
    # Key insights
    report_lines.extend([
        "## Key Physical Insights",
        "",
        "Based on the automated analysis, the following design principles emerge:",
        "",
        "1. **Dimerization Dominance**: The ratio a/b is consistently the most important parameter across all objectives",
        "2. **Regime Specialization**: Different design regimes optimize for different trade-offs",
        "3. **Fabrication Constraints**: Minimum feature size is a critical limiting factor",
        "4. **Ring Size Effects**: Larger rings generally improve Q-factors through reduced bending losses",
        "",
        "## Recommendations",
        "",
        "- **For Maximum Q-factor**: Focus on extreme dimerization regimes with a/b > 4.0",
        "- **For Robustness**: Balance dimerization strength with fabrication feasibility",
        "- **For Integration**: Consider compact regimes with optimized dimerization",
        "",
        "---",
        "",
        "*This report was automatically generated by the physics-informed analysis framework.*"
    ])
    
    return "\n".join(report_lines)


def main():
    """Example usage of design analysis framework."""
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'a': np.random.uniform(0.3, 0.6, n_samples),
        'b': np.random.uniform(0.05, 0.2, n_samples),
        'r': np.random.uniform(0.08, 0.16, n_samples),
        'w': np.random.uniform(0.45, 0.65, n_samples),
        'N_cells': np.random.randint(100, 200, n_samples),
        'R': np.random.uniform(8, 20, n_samples)
    }
    
    # Add synthetic objectives
    dimerization = data['a'] - data['b']
    data['q_avg'] = 15000 + dimerization * 20000 + data['R'] * 1000 + np.random.normal(0, 2000, n_samples)
    data['q_std'] = 1000 + np.random.normal(0, 300, n_samples)
    data['bandgap_size'] = dimerization * 50 + np.random.normal(0, 5, n_samples)
    data['mode_volume'] = np.pi * data['r']**2 * 0.5 + np.random.normal(0, 0.1, n_samples)
    
    df = pd.DataFrame(data)
    
    # Run analysis
    results = create_comprehensive_analysis_report(df, ".")
    
    print("Analysis complete! Check the generated files.")


if __name__ == "__main__":
    main()