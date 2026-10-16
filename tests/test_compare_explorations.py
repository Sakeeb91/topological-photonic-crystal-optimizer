"""
Tests for compare_explorations.py run labeling
"""
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compare_explorations import config_label


def _write_run_config(tmp_path, run_info):
    (tmp_path / 'run_config.yaml').write_text(yaml.dump({'seed': 1, 'run_info': run_info}))


def test_known_config_gets_its_label(tmp_path):
    _write_run_config(tmp_path, {'config_file': 'explore_large_rings.yaml'})
    assert config_label(str(tmp_path)) == 'Large Rings'


def test_unknown_config_uses_file_stem(tmp_path):
    _write_run_config(tmp_path, {'config_file': 'my_sweep.yaml'})
    assert config_label(str(tmp_path)) == 'my_sweep'


def test_runs_without_config_record_return_none(tmp_path):
    assert config_label(str(tmp_path)) is None           # no run_config.yaml
    _write_run_config(tmp_path, {'timestamp': 'x'})
    assert config_label(str(tmp_path)) is None           # saved before config_file existed
