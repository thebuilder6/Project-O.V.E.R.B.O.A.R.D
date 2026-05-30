"""
Test runner for FLL Trajectory Optimizer test suite.

Runs all tests and generates a coverage report.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test modules
from tests.test_jax_heuristics import TestJAXHeuristics
from tests.test_refinement_accuracy import TestRefinementAccuracy
from tests.test_critic_metrics import TestCriticMetrics


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestJAXHeuristics))
    suite.addTests(loader.loadTestsFromTestCase(TestRefinementAccuracy))
    suite.addTests(loader.loadTestsFromTestCase(TestCriticMetrics))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\nError tests:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
