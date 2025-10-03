#!/usr/bin/env python3
"""
Test runner script to execute unit tests with proper environment setup
"""

import os
import sys
import subprocess


def run_tests():
    """Run the unit tests with proper environment setup"""

    # Change to src directory to resolve relative imports
    src_dir = os.path.join(os.path.dirname(__file__), 'src')
    tests_dir = os.path.join(os.path.dirname(__file__), 'tests')

    # Set PYTHONPATH to src directory
    env = os.environ.copy()
    env['PYTHONPATH'] = src_dir

    # Change to src directory
    os.chdir(src_dir)

    # Run pytest
    cmd = [
        sys.executable, '-m', 'pytest',
        tests_dir,
        '-v',
        '--tb=short',
        '--durations=10'
    ]

    print("Running unit tests...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {src_dir}")
    print(f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
