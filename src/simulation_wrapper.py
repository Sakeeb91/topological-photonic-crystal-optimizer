import numpy as np
import time
# TODO: Uncomment the following line when you are ready for real simulations
# import meep as mp

# A very large negative number for failed simulations
_NEGINF = -1.0e10

def _calculate_objective(q_factors, config):
    """Calculates the final score from a list of Q-factors."""
    if not q_factors:
        return _NEGINF

    q_avg = np.mean(q_factors)
    q_std = np.std(q_factors)
    
    penalty_factor = config['objective']['q_penalty_factor']
    score = q_avg - penalty_factor * q_std
    
    return score

def evaluate_design_mock(design_vector, config):
    """
    MOCK FUNCTION: Simulates the performance of a design without running MEEP.
    This is for testing the optimization loop quickly.
    It returns a higher score for designs that are 'better' based on a simple formula.
    """
    # Unpack design vector for clarity
    a, b, r, R, w = design_vector
    
    print(f"  [Mock Sim] Evaluating: a={a:.3f}, b={b:.3f}, r={r:.3f}, R={R:.2f}, w={w:.3f}")

    # --- Let's create a fake physics model ---
    # Goal: A large (a-b) difference is good (strong dimerization)
    # Goal: A larger R is good (less bending loss)
    # Goal: Radius 'r' has an optimal value around 0.15
    ideal_dimerization = (a - b) * 1e5
    ideal_radius = -((r - 0.15)**2) * 1e5
    ideal_ring_radius = R * 1000

    base_q = 20000 + ideal_dimerization + ideal_radius + ideal_ring_radius
    
    # Simulate disorder
    num_runs = config['objective']['num_disorder_runs']
    # The mock 'disorder' makes the Q-factor noisy
    q_factors = base_q + np.random.randn(num_runs) * 2000 
    
    time.sleep(0.1) # Simulate that the function takes some time
    
    score = _calculate_objective(q_factors, config)
    print(f"  [Mock Sim] Result -> Avg Q: {np.mean(q_factors):.0f}, Std Q: {np.std(q_factors):.0f}, Score: {score:.2f}")
    
    return score


def evaluate_design_meep(design_vector, config):
    """
    REAL FUNCTION: Evaluates a design by running MEEP FDTD simulations.
    
    This is the function you will build out.
    """
    # TODO: This is your main task.
    print("  [MEEP Sim] This function is not implemented. Returning a random score.")
    # You would replace this with your MEEP script logic.
    # The logic would look something like this:
    
    # 1. Unpack design_vector and config parameters
    a, b, r, R, w = design_vector
    resolution = config['simulation']['resolution']
    # ... and so on
    
    q_factors = []
    num_runs = config['objective']['num_disorder_runs']
    disorder_std = r * (config['objective']['disorder_std_dev_percent'] / 100.0)

    # for i in range(num_runs):
    #     # 2. Define MEEP geometry. This is the complex part.
    #     # You need to programmatically generate the ring of holes.
    #     # For each hole, its radius would be r + np.random.randn() * disorder_std
    #     
    #     # 3. Define source and simulation object
    #     # e.g., src = mp.Source(...)
    #     
    #     # 4. Use Harminv to find the Q-factor
    #     # harminv_instance = mp.Harminv(...)
    #     # sim = mp.Simulation(...)
    #     # sim.run(until_after_sources=...)
    #     
    #     # 5. Extract Q from harminv_instance and append to q_factors list
    #     # q_val = harminv_instance.modes[0].Q # (handle cases with no modes found)
    #     # q_factors.append(q_val)
    
    # For now, we return a random score to make the optimizer run.
    # Replace this with the real calculation.
    mock_q = np.random.rand() * 50000
    return mock_q