# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rohith Reddy Vennam, Luke Wilson, Ish Kumar Jain, Dinesh Bharadia
# UC San Diego Wireless Communications Sensing and Networking Group (WCSNG)
"""
Smoke tests: each script runs end-to-end with --quick and exits without error.

These do NOT check figure correctness — they verify the import chain and
code paths execute without crashing on tiny inputs.
"""
import subprocess
import sys
import os
import pytest

PYTHON = sys.executable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCRIPTS_DIR = os.path.join(REPO_ROOT, 'scripts')


def run_script(name, extra_args=None):
    script = os.path.join(SCRIPTS_DIR, name)
    cmd = [PYTHON, script, '--quick']
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    env['PYTHONPATH'] = REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if result.returncode != 0:
        pytest.fail(
            f"{name} failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


class TestSmokeScripts:
    def test_fig02_parabolic_gain(self):
        run_script("fig02_parabolic_gain.py")

    def test_fig04_gain_vs_arrays(self):
        run_script("fig04_gain_vs_arrays.py")

    def test_fig06_mimo_boundaries(self):
        run_script("fig06_mimo_boundaries.py")

    def test_fig09_beampattern_sim(self):
        run_script("fig09_beampattern_sim.py")

    def test_fig10_hardware_validation(self):
        # Hardware data is not expected in CI — quick mode still runs theory+sim
        run_script("fig10_hardware_validation.py")

    def test_fig11_2d_beampattern(self):
        run_script("fig11_2d_beampattern.py")

    def test_fig12_mimo_dof(self):
        run_script("fig12_mimo_dof.py")

    def test_fig13_throughput(self):
        run_script("fig13_throughput.py")
