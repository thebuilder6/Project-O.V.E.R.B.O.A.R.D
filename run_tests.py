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

# Import test modules (non-JAX dependent)
from tests.test_optimizer import (
    TestTrajectoryOptimizer,
    TestFastPolish,
    TestSolverIterationBudget,
    TestMergeConditions,
    TestBenchmarkDryRun,
    TestSelfConsistentPlan,
)
from tests.test_robot_model import (
    TestRobotConfig,
    TestDifferentialDriveModel,
    TestMeasuredMotorConfig,
)
from tests.test_validator import TestForwardIntegrate, TestAuditConstraints, TestComputeMetrics, TestValidateTrajectory
from tests.test_random_suite import (
    TestRandomSuiteGenerator,
)
from tests.test_sysid_fit import (
    TestSysIdFit,
    TestNoLoadSpeed,
    TestDrivebaseProfile,
    TestHubSessionPoses,
)
from tests.test_digital_twin import (
    TestDigitalTwinConfig,
    TestDigitalTwinValidator,
    TestMeasuredMotorFit,
    TestMeasuredMotorEndToEnd,
    TestDigitalTwinMechanisms,
)
from tests.test_fields import (
    TestObstacleOverlap,
    TestFieldCollisions,
    TestFieldFiles,
    TestServerPayload,
    TestLaunchAreas,
    TestRoboVibeField,
    TestRoboVibeConverter,
    TestMatDownload,
)
from tests.test_mat_import import (
    TestAutocrop,
    TestComposeMat,
    TestPdfRendering,
)
from tests.test_drivetrain import (
    TestMotorLaw,
    TestDrivetrainModel,
    TestDeclaredMotorCheck,
)
from tests.test_validator_model import TestBacklashModel, TestHeadingMetric
from tests.test_sysid_compare import TestSysIdCompare
from tests.test_export_compact import TestCompactExport
from tests.test_export import TestResampleHeading
from tests.test_hub_deploy import (
    TestSessionFile,
    TestPushHubCli,
    TestFirmwareCompiles,
    TestHubRunnerContract,
)
from tests.test_sysid_collect import (
    TestCollectorMetadata,
    TestSessionPoses,
    TestFittedScales,
    TestMeasurementSheet,
    TestRepeatStatistics,
    TestRefinedStepFit,
    TestDriftRegression,
    TestDriftAndStallTorque,
    TestWallHoming,
    TestCollectCommand,
)
from tests.test_firmware_events import (
    TestFinalWaypointEvent,
    TestFinalWaypointEventCompact,
    TestFirmwareParity,
    TestLateTickEvents,
)
from tests.test_cli_ux import (
    TestCliBasics,
    TestStudioEntryPoints,
    TestParallelCandidates,
    TestValidationReport,
    TestIterationHistory,
)
from tests.test_gui_server import (
    TestStudioRoutes,
    TestStudioSolve,
    TestStudioPush,
    TestStudioModel,
    TestStudioCalibration,
    TestStudioHelpers,
    TestStudioReport,
)
from tests.test_studio_inventory import (
    TestStudioInventory,
)
from tests.test_studio_report import (
    TestReportBuilder,
    TestProjectFile,
    TestStudioReportRoute,
    TestStudioInventoryNewControls,
)
from tests.test_studio_honesty import (
    TestTheHiddenAttributeWins,
    TestThePushNeedsAValidatedTrajectory,
    TestAFailedSolveLeavesNoPath,
    TestASolveThatNeverStartsClearsTheVerdict,
    TestTheMessageTheOperatorReads,
    TestTheResultExplainsItself,
    TestTheSheetWithoutASession,
)

# Regression tests (no JAX required)
from tests.test_regressions import (
    TestExportConfig,
    TestAttemptSelection,
    TestCandidateSpecs,
    TestReachabilityEnvelope,
    TestBenchmarkValidationConfig,
    TestExportGate,
    TestValidatorGates,
    TestWriteGateIntegration,
    TestOpenLoopModel,
    TestSweepSpecs,
)

# JAX-dependent tests (skip if jax is not installed)
try:
    from tests.test_jax_heuristics import TestJAXHeuristics
    from tests.test_jax_ramsete import TestJAXRamsete, TestJAXDriveQuantization
    from tests.test_refinement_accuracy import TestRefinementAccuracy
    from tests.test_critic_metrics import TestCriticMetrics
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: JAX-dependent tests skipped ({e})")
    JAX_AVAILABLE = False


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    if JAX_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestJAXHeuristics))
        suite.addTests(loader.loadTestsFromTestCase(TestJAXRamsete))
        suite.addTests(loader.loadTestsFromTestCase(TestJAXDriveQuantization))
        suite.addTests(loader.loadTestsFromTestCase(TestRefinementAccuracy))
        suite.addTests(loader.loadTestsFromTestCase(TestCriticMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestExportConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestAttemptSelection))
    suite.addTests(loader.loadTestsFromTestCase(TestCandidateSpecs))
    suite.addTests(loader.loadTestsFromTestCase(TestReachabilityEnvelope))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkValidationConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestExportGate))
    suite.addTests(loader.loadTestsFromTestCase(TestValidatorGates))
    suite.addTests(loader.loadTestsFromTestCase(TestWriteGateIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenLoopModel))
    suite.addTests(loader.loadTestsFromTestCase(TestSweepSpecs))
    suite.addTests(loader.loadTestsFromTestCase(TestTrajectoryOptimizer))
    suite.addTests(loader.loadTestsFromTestCase(TestRobotConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDifferentialDriveModel))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasuredMotorConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestForwardIntegrate))
    suite.addTests(loader.loadTestsFromTestCase(TestAuditConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestValidateTrajectory))
    suite.addTests(loader.loadTestsFromTestCase(TestSysIdFit))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomSuiteGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestNoLoadSpeed))
    suite.addTests(loader.loadTestsFromTestCase(TestDrivebaseProfile))
    suite.addTests(loader.loadTestsFromTestCase(TestHubSessionPoses))
    suite.addTests(loader.loadTestsFromTestCase(TestDigitalTwinConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestDigitalTwinValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasuredMotorFit))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasuredMotorEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestDigitalTwinMechanisms))
    suite.addTests(loader.loadTestsFromTestCase(TestMotorLaw))
    suite.addTests(loader.loadTestsFromTestCase(TestDrivetrainModel))
    suite.addTests(loader.loadTestsFromTestCase(TestDeclaredMotorCheck))
    suite.addTests(loader.loadTestsFromTestCase(TestObstacleOverlap))
    suite.addTests(loader.loadTestsFromTestCase(TestFieldCollisions))
    suite.addTests(loader.loadTestsFromTestCase(TestFieldFiles))
    suite.addTests(loader.loadTestsFromTestCase(TestServerPayload))
    suite.addTests(loader.loadTestsFromTestCase(TestLaunchAreas))
    suite.addTests(loader.loadTestsFromTestCase(TestRoboVibeField))
    suite.addTests(loader.loadTestsFromTestCase(TestRoboVibeConverter))
    suite.addTests(loader.loadTestsFromTestCase(TestMatDownload))
    suite.addTests(loader.loadTestsFromTestCase(TestAutocrop))
    suite.addTests(loader.loadTestsFromTestCase(TestComposeMat))
    suite.addTests(loader.loadTestsFromTestCase(TestPdfRendering))
    suite.addTests(loader.loadTestsFromTestCase(TestSysIdCompare))
    suite.addTests(loader.loadTestsFromTestCase(TestCompactExport))
    suite.addTests(loader.loadTestsFromTestCase(TestResampleHeading))
    suite.addTests(loader.loadTestsFromTestCase(TestFinalWaypointEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestFinalWaypointEventCompact))
    suite.addTests(loader.loadTestsFromTestCase(TestFirmwareParity))
    suite.addTests(loader.loadTestsFromTestCase(TestLateTickEvents))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionFile))
    suite.addTests(loader.loadTestsFromTestCase(TestPushHubCli))
    suite.addTests(loader.loadTestsFromTestCase(TestFirmwareCompiles))
    suite.addTests(loader.loadTestsFromTestCase(TestHubRunnerContract))
    suite.addTests(loader.loadTestsFromTestCase(TestCollectorMetadata))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionPoses))
    suite.addTests(loader.loadTestsFromTestCase(TestFittedScales))
    suite.addTests(loader.loadTestsFromTestCase(TestMeasurementSheet))
    suite.addTests(loader.loadTestsFromTestCase(TestRepeatStatistics))
    suite.addTests(loader.loadTestsFromTestCase(TestRefinedStepFit))
    suite.addTests(loader.loadTestsFromTestCase(TestDriftRegression))
    suite.addTests(loader.loadTestsFromTestCase(TestDriftAndStallTorque))
    suite.addTests(loader.loadTestsFromTestCase(TestWallHoming))
    suite.addTests(loader.loadTestsFromTestCase(TestCollectCommand))
    suite.addTests(loader.loadTestsFromTestCase(TestCliBasics))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioEntryPoints))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioPush))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioModel))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioCalibration))
    suite.addTests(loader.loadTestsFromTestCase(TestBacklashModel))
    suite.addTests(loader.loadTestsFromTestCase(TestHeadingMetric))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelCandidates))
    suite.addTests(loader.loadTestsFromTestCase(TestValidationReport))
    suite.addTests(loader.loadTestsFromTestCase(TestIterationHistory))
    suite.addTests(loader.loadTestsFromTestCase(TestFastPolish))
    suite.addTests(loader.loadTestsFromTestCase(TestSolverIterationBudget))
    suite.addTests(loader.loadTestsFromTestCase(TestMergeConditions))
    suite.addTests(loader.loadTestsFromTestCase(TestBenchmarkDryRun))
    suite.addTests(loader.loadTestsFromTestCase(TestSelfConsistentPlan))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioRoutes))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioSolve))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioHelpers))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioInventory))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioReport))
    suite.addTests(loader.loadTestsFromTestCase(TestReportBuilder))
    suite.addTests(loader.loadTestsFromTestCase(TestProjectFile))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioReportRoute))
    suite.addTests(loader.loadTestsFromTestCase(TestStudioInventoryNewControls))
    suite.addTests(loader.loadTestsFromTestCase(TestTheHiddenAttributeWins))
    suite.addTests(loader.loadTestsFromTestCase(TestThePushNeedsAValidatedTrajectory))
    suite.addTests(loader.loadTestsFromTestCase(TestAFailedSolveLeavesNoPath))
    suite.addTests(loader.loadTestsFromTestCase(TestASolveThatNeverStartsClearsTheVerdict))
    suite.addTests(loader.loadTestsFromTestCase(TestTheMessageTheOperatorReads))
    suite.addTests(loader.loadTestsFromTestCase(TestTheResultExplainsItself))
    suite.addTests(loader.loadTestsFromTestCase(TestTheSheetWithoutASession))
    
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
