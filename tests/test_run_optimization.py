"""
End-to-end tests for run_optimization.py on a tiny seeded config
"""
import glob
import os
import sys

import pandas as pd
import pytest
import yaml

REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, REPO_ROOT)

import run_optimization

FAILED_EVALUATION = 1.0e9  # evaluate_design returns -1e10 when every disorder run fails


@pytest.fixture(scope='module')
def tiny_run(tmp_path_factory):
    """Run a 5-evaluation optimization in a temp dir and return its results dir"""
    tmp_path = tmp_path_factory.mktemp('run')
    with open(os.path.join(REPO_ROOT, 'configs', 'strong_dimerization_v1.yaml')) as f:
        config = yaml.safe_load(f)
    config['seed'] = 7
    config['objective']['num_disorder_runs'] = 3
    config['optimizer']['n_initial_points'] = 3
    config['optimizer']['n_iterations'] = 2

    # yaml.dump sorts keys, so design_space is written as R, a, b, r, w; the same
    # happens to the run_config.yaml saved with every run
    config_path = tmp_path / 'tiny.yaml'
    config_path.write_text(yaml.dump(config))

    cwd = os.getcwd()
    os.chdir(tmp_path)  # results/ is created relative to the cwd
    try:
        run_optimization.main(str(config_path))
    finally:
        os.chdir(cwd)

    (results_dir,) = glob.glob(str(tmp_path / 'results' / 'run_*'))
    log = pd.read_csv(os.path.join(results_dir, 'optimization_log.csv'))
    with open(os.path.join(results_dir, 'best_params.yaml')) as f:
        best_params = yaml.safe_load(f)
    return log, best_params


@pytest.mark.integration
def test_design_space_key_order_does_not_matter(tiny_run):
    """Every evaluation succeeds even when design_space keys are not in a, b, r, R, w order"""
    log, _ = tiny_run
    assert len(log) == 5
    assert (log['score'].abs() < FAILED_EVALUATION).all()
