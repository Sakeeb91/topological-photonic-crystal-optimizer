"""
Simulation Wrapper for Topological Photonic Crystal Ring Resonators

This module provides both mock and MEEP-based simulation functions for evaluating
SSH ring resonator designs.

MEEP INTEGRATION STATUS:
========================
Currently, the MEEP integration is PARTIALLY IMPLEMENTED. The code structure and
simulation workflow are complete, but MEEP execution is replaced with a physics-based
mock to allow the optimization framework to function without MEEP installation.

To enable full MEEP simulations:
1. Install MEEP: conda install -c conda-forge pymeep
2. Uncomment the MEEP import below
3. Uncomment the MEEP simulation code in evaluate_design_meep()
4. Remove or modify the _simulate_physics_model() fallback

For most development and testing purposes, the mock simulation provides realistic
behavior and is much faster than full electromagnetic simulations.
"""

import numpy as np
import time
import math

# MEEP Integration - Currently disabled for mock operation
# Uncomment when MEEP is installed and you want full EM simulations
# import meep as mp
MEEP_AVAILABLE = False  # Set to True when MEEP is installed and imported

# A very large negative number for failed simulations
_NEGINF = -1.0e10

def _calculate_objective(q_factors, config):
    """Calculates the final score from a list of Q-factors."""
    if len(q_factors) == 0:
        return _NEGINF

    q_avg = np.mean(q_factors)
    q_std = np.std(q_factors)
    
    penalty_factor = config['objective']['q_penalty_factor']
    score = q_avg - penalty_factor * q_std
    
    return score

def _calculate_comprehensive_objectives(q_factors, bandgaps, mode_volumes, design_vector, config):
    """
    Calculate comprehensive objectives for multi-objective optimization.
    
    Args:
        q_factors: List of Q-factors from disorder runs
        bandgaps: List of bandgap sizes
        mode_volumes: List of mode volumes
        design_vector: [a, b, r, R, w]
        config: Configuration dictionary
    
    Returns:
        Dictionary with all objectives
    """
    if len(q_factors) == 0:
        return {
            'q_factor': 0,
            'q_std': 1e6,
            'bandgap_size': 0,
            'mode_volume': 1e6,
            'score': _NEGINF
        }
    
    a, b, r, R, w = design_vector
    
    # Primary objectives
    q_avg = np.mean(q_factors)
    q_std = np.std(q_factors) if len(q_factors) > 1 else 0
    bandgap_avg = np.mean(bandgaps) if bandgaps else abs(a - b)  # Fallback to dimerization
    mode_volume_avg = np.mean(mode_volumes) if mode_volumes else np.pi * r**2  # Fallback approximation
    
    # Legacy score for backward compatibility
    penalty_factor = config['objective'].get('q_penalty_factor', 1.0)
    score = q_avg - penalty_factor * q_std
    
    return {
        'q_factor': q_avg,
        'q_std': q_std,
        'bandgap_size': bandgap_avg,
        'mode_volume': mode_volume_avg,
        'score': score,
        'dimerization_strength': abs(a - b),
        'robustness_ratio': q_avg / max(q_std, 1e-6)
    }

def evaluate_design_mock(design_vector, config):
    """
    Enhanced MOCK FUNCTION: Simulates comprehensive performance metrics based on thesis insights.
    
    Models the key trade-off between lattice confinement and radiation confinement
    as identified in the AlexisHK thesis analysis.
    """
    # Unpack design vector for clarity
    a, b, r, R, w = design_vector
    
    print(f"  [Mock Sim] Evaluating: a={a:.3f}, b={b:.3f}, r={r:.3f}, R={R:.2f}, w={w:.3f}")

    # --- Enhanced physics model based on thesis insights ---
    
    # Dimerization strength (primary factor for topological protection)
    dimerization = abs(a - b)
    dimerization_ratio = a / max(b, 1e-6)
    
    # Model the trade-off: lattice confinement vs radiation confinement
    # Thesis insight: larger bandgap doesn't always mean higher Q-factor
    
    # Lattice confinement factor (benefits from moderate dimerization)
    lattice_confinement = np.exp(-((dimerization - 0.15)**2) / (2 * 0.05**2))
    
    # Radiation confinement factor (benefits from large ring, optimal hole size)
    bending_loss_factor = np.exp(-5.0 / R)  # Exponential improvement with R
    hole_coupling_factor = np.exp(-((r - 0.12)**2) / (2 * 0.03**2))  # Optimal around 0.12
    radiation_confinement = (1 - bending_loss_factor) * hole_coupling_factor
    
    # Base Q-factor model incorporating trade-off
    base_q = 15000 + lattice_confinement * 20000 + radiation_confinement * 25000
    
    # Bandgap size model (increases with dimerization)
    bandgap_base = dimerization * 50 + np.random.normal(0, 2)  # Units arbitrary
    
    # Mode volume model (roughly scales with hole area)
    mode_volume_base = np.pi * r**2 * 0.5 + 0.1  # Approximate
    
    # Simulate disorder runs
    num_runs = config['objective']['num_disorder_runs']

    q_factors = []
    bandgaps = []
    mode_volumes = []
    failed_runs = 0

    for i in range(num_runs):
        try:
            # Add disorder to each metric
            q_disorder = np.random.normal(0, 1500 * (1 + dimerization * 2))  # More stable with higher dimerization
            bandgap_disorder = np.random.normal(0, 1.0)
            mode_vol_disorder = np.random.normal(0, 0.02)

            q_factors.append(max(1000, base_q + q_disorder))
            bandgaps.append(max(0.1, bandgap_base + bandgap_disorder))
            mode_volumes.append(max(0.05, mode_volume_base + mode_vol_disorder))
        except Exception as e:
            failed_runs += 1
            print(f"    WARNING: Mock disorder run {i+1} failed: {e}")
    
    time.sleep(0.1)  # Simulate computation time

    # Check if we have enough successful runs
    if len(q_factors) == 0:
        print(f"  [Mock Sim] ERROR: All {num_runs} disorder runs failed")
        return _NEGINF if not config.get('return_comprehensive_objectives', False) else {
            'q_factor': 0, 'q_std': 1e6, 'bandgap_size': 0,
            'mode_volume': 1e6, 'score': _NEGINF
        }

    if failed_runs > 0:
        print(f"  [Mock Sim] Note: {failed_runs}/{num_runs} runs failed, using {len(q_factors)} results")

    # Calculate comprehensive objectives
    objectives = _calculate_comprehensive_objectives(q_factors, bandgaps, mode_volumes, design_vector, config)
    
    print(f"  [Mock Sim] Q: {objectives['q_factor']:.0f}±{objectives['q_std']:.0f}, "
          f"Bandgap: {objectives['bandgap_size']:.2f}, ModeVol: {objectives['mode_volume']:.3f}")
    
    # Return full objectives for multi-objective optimization, legacy score for backward compatibility
    return_comprehensive = config.get('return_comprehensive_objectives', False)
    if return_comprehensive:
        return objectives
    else:
        return objectives['score']  # Legacy behavior


def _generate_ssh_ring_geometry(a, b, r, R, w, disorder_std=0.0):
    """
    Generate geometry for SSH (Su-Schrieffer-Heeger) ring resonator.
    
    Parameters:
    - a, b: dimerization distances
    - r: nominal hole radius
    - R: ring radius 
    - w: waveguide width
    - disorder_std: standard deviation for hole radius disorder
    
    Returns:
    - List of hole positions and radii [(x, y, radius), ...]
    """
    holes = []
    
    # Calculate number of unit cells that fit around the ring
    unit_cell_length = a + b
    circumference = 2 * np.pi * R
    num_unit_cells = int(circumference / unit_cell_length)
    
    # Adjust unit cell length to fit exactly around the ring
    actual_unit_length = circumference / num_unit_cells
    actual_a = a * (actual_unit_length / unit_cell_length)
    actual_b = b * (actual_unit_length / unit_cell_length)
    
    current_angle = 0.0
    
    for cell_idx in range(num_unit_cells):
        # First hole of the unit cell (spacing 'a')
        hole_radius = r + np.random.normal(0, disorder_std)
        hole_radius = max(hole_radius, 0.01)  # Minimum radius
        
        x = R * np.cos(current_angle)
        y = R * np.sin(current_angle)
        holes.append((x, y, hole_radius))
        
        # Move to next hole position by distance 'a'
        angle_increment = actual_a / R
        current_angle += angle_increment
        
        # Second hole of the unit cell (spacing 'b')
        hole_radius = r + np.random.normal(0, disorder_std)
        hole_radius = max(hole_radius, 0.01)  # Minimum radius
        
        x = R * np.cos(current_angle)
        y = R * np.sin(current_angle)
        holes.append((x, y, hole_radius))
        
        # Move to next unit cell by distance 'b'
        angle_increment = actual_b / R
        current_angle += angle_increment
    
    return holes, num_unit_cells

def _create_meep_geometry(design_vector, disorder_std, config):
    """
    Create MEEP geometry objects for the SSH ring resonator.
    
    Returns:
    - geometry: List of MEEP geometry objects
    - cell_size: MEEP cell size
    """
    # This is a placeholder for when MEEP is available
    # import meep as mp
    
    a, b, r, R, w = design_vector
    
    # Generate hole positions
    holes, num_cells = _generate_ssh_ring_geometry(a, b, r, R, w, disorder_std)
    
    # Calculate simulation cell size (needs padding for PML)
    pml_width = config['simulation']['pml_width']
    cell_size = 2 * (R + w/2) + 4 * pml_width
    
    # Create geometry objects (placeholder)
    geometry = []
    
    # Substrate (silicon dioxide, n≈1.46)
    # substrate = mp.Block(material=mp.Medium(index=1.46),
    #                     size=mp.Vector3(cell_size, cell_size, mp.inf))
    # geometry.append(substrate)
    
    # Ring waveguide (silicon nitride, n≈2.0)
    # ring_material = mp.Medium(index=2.0)
    # 
    # # Create ring by subtracting inner cylinder from outer cylinder
    # outer_ring = mp.Cylinder(radius=R + w/2, material=ring_material)
    # inner_ring = mp.Cylinder(radius=R - w/2, material=mp.Medium(index=1.0))
    # geometry.extend([outer_ring, inner_ring])
    
    # Add holes
    # for x, y, hole_r in holes:
    #     hole = mp.Cylinder(center=mp.Vector3(x, y, 0), 
    #                       radius=hole_r,
    #                       material=mp.Medium(index=1.0))  # Air holes
    #     geometry.append(hole)
    
    return geometry, cell_size, holes

def evaluate_design_meep(design_vector, config):
    """
    MEEP FDTD Simulation Function (Currently using physics-based mock)

    This function implements the complete MEEP simulation workflow for
    evaluating SSH ring resonator designs with disorder robustness.

    NOTE: MEEP execution is currently replaced with a physics-based model
    (see _simulate_physics_model) until MEEP is fully installed and enabled.
    The simulation structure and disorder loop are production-ready.

    Args:
        design_vector: [a, b, r, R, w] design parameters
        config: Configuration dictionary with simulation settings

    Returns:
        float: Robustness score (Q_avg - penalty_factor * Q_std)
    """
    try:
        # import meep as mp
        if MEEP_AVAILABLE:
            print("  [MEEP Sim] Starting MEEP FDTD simulation...")
        else:
            print("  [MEEP Sim] Using physics-based mock (MEEP not available)...")
        
        # 1. Unpack design_vector and config parameters
        a, b, r, R, w = design_vector
        resolution = config['simulation']['resolution']
        sim_time = config['simulation']['sim_time']
        target_wavelength = config['simulation']['target_wavelength']
        pml_width = config['simulation']['pml_width']
        
        # 2. Disorder parameters
        num_runs = config['objective']['num_disorder_runs']
        disorder_std = r * (config['objective']['disorder_std_dev_percent'] / 100.0)
        
        q_factors = []
        failed_runs = 0
        min_successful_runs = max(1, num_runs // 2)  # Require at least half to succeed

        print(f"  [MEEP Sim] Running {num_runs} disorder simulations...")
        print(f"  [MEEP Sim] Parameters: a={a:.3f}, b={b:.3f}, r={r:.3f}, R={R:.2f}, w={w:.3f}")

        for run_idx in range(num_runs):
            try:
                print(f"    Disorder run {run_idx + 1}/{num_runs}")

                # 3. Generate geometry with disorder
                geometry, cell_size, holes = _create_meep_geometry(design_vector, disorder_std, config)
            
            # Since MEEP is not actually imported, we'll simulate the process
            # In real implementation, this would be:
            #
            # # 4. Set up MEEP simulation
            # cell = mp.Vector3(cell_size, cell_size, 0)
            # 
            # # Perfectly Matched Layer boundary conditions
            # pml_layers = [mp.PML(thickness=pml_width)]
            # 
            # # Source: Gaussian pulse to excite cavity modes
            # source_freq = 1/target_wavelength  # Convert wavelength to frequency
            # source_width = 0.1 * source_freq   # Pulse width
            # 
            # # Place source near the ring to couple into whispering gallery modes
            # source_pos = mp.Vector3(R * 0.9, 0, 0)
            # sources = [mp.Source(mp.GaussianSource(frequency=source_freq, fwidth=source_width),
            #                     component=mp.Hz,
            #                     center=source_pos)]
            # 
            # # 5. Set up Harminv for mode analysis
            # # Monitor point inside the cavity
            # monitor_pos = mp.Vector3(R * 0.95, 0, 0)
            # harminv = mp.Harminv(mp.Hz, monitor_pos, source_freq, source_width)
            # 
            # # 6. Create and run simulation
            # sim = mp.Simulation(cell_size=cell,
            #                    boundary_layers=pml_layers,
            #                    geometry=geometry,
            #                    sources=sources,
            #                    resolution=resolution)
            # 
            # sim.run(mp.at_beginning(mp.output_epsilon),
            #        mp.after_sources(harminv),
            #        until_after_sources=sim_time)
            # 
            # # 7. Extract Q-factor from Harminv results
            # if len(harminv.modes) > 0:
            #     # Find mode closest to target frequency
            #     best_mode = None
            #     min_freq_diff = float('inf')
            #     
            #     for mode in harminv.modes:
            #         freq_diff = abs(mode.freq - source_freq)
            #         if freq_diff < min_freq_diff and mode.Q > 100:  # Quality filter
            #             min_freq_diff = freq_diff
            #             best_mode = mode
            #     
            #     if best_mode is not None:
            #         q_factor = abs(best_mode.Q)
            #         print(f"      Found mode: freq={best_mode.freq:.4f}, Q={q_factor:.0f}")
            #         q_factors.append(q_factor)
            #     else:
            #         print(f"      No suitable modes found")
            #         q_factors.append(1000)  # Low Q penalty
            # else:
            #     print(f"      No modes detected")
            #     q_factors.append(1000)  # Low Q penalty
            
                # For now, simulate the MEEP results with physics-based model
                # This maintains the disorder loop structure while MEEP is not available
                base_q = _simulate_physics_model(a, b, r, R, w, holes)
                simulated_q = max(1000, base_q + np.random.normal(0, base_q * 0.1))
                q_factors.append(simulated_q)

                print(f"      Simulated Q-factor: {simulated_q:.0f}")

            except Exception as run_error:
                failed_runs += 1
                print(f"      WARNING: Disorder run {run_idx + 1} failed: {run_error}")
                # Continue to next run instead of failing entire evaluation

        # Check if we have enough successful runs
        if len(q_factors) < min_successful_runs:
            print(f"  [MEEP Sim] Failed - only {len(q_factors)}/{num_runs} runs succeeded "
                  f"(minimum {min_successful_runs} required)")
            return _NEGINF

        if failed_runs > 0:
            print(f"  [MEEP Sim] Note: {failed_runs}/{num_runs} runs failed, "
                  f"using {len(q_factors)} successful results")

        # 8. Calculate final robustness score
        if len(q_factors) > 0:
            score = _calculate_objective(q_factors, config)
            print(f"  [MEEP Sim] Completed. Avg Q: {np.mean(q_factors):.0f}, "
                  f"Std Q: {np.std(q_factors):.0f}, Score: {score:.2f}")
            return score
        else:
            print(f"  [MEEP Sim] Failed - no valid Q-factors obtained")
            return _NEGINF
            
    except Exception as e:
        print(f"  [MEEP Sim] Error: {e}")
        return _NEGINF

def _simulate_physics_model(a, b, r, R, w, holes):
    """
    Physics-based model to simulate Q-factor until MEEP is fully integrated.
    This provides realistic behavior for testing the optimization framework.
    """
    # Base Q from ring parameters (larger R = less bending loss)
    base_q = 20000 + R * 2000
    
    # Dimerization factor (stronger dimerization = better topological protection)
    dimerization = (a - b) / (a + b) if (a + b) > 0 else 0
    dimerization_bonus = dimerization * 15000
    
    # Hole coupling (optimal radius around 0.15 for this wavelength)
    optimal_r = 0.15
    hole_penalty = -((r - optimal_r)**2) * 50000
    
    # Fabrication realism (too small features are hard to make)
    if r < 0.08 or w < 0.4:
        fabrication_penalty = -10000
    else:
        fabrication_penalty = 0
    
    # Disorder impact (more holes = more sensitive to disorder)
    num_holes = len(holes)
    disorder_penalty = -num_holes * 100
    
    total_q = base_q + dimerization_bonus + hole_penalty + fabrication_penalty + disorder_penalty
    return max(1000, total_q)