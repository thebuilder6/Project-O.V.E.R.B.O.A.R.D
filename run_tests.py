"""
Test runner for FLL Trajectory Optimizer test suite.

Runs all tests and generates a coverage report.

The runner keeps a cache of the tests that passed. A test whose code and
data have not changed since it passed is skipped, so a second run over an
unchanged tree costs a fraction of the first one. `testcache.py` holds the
rules; `python run_tests.py --cache-status` says what the cache would do and
`python run_tests.py --no-cache` ignores it and runs everything.
"""

import sys
import time
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import testcache

# Import test modules (non-JAX dependent)
from tests.test_optimizer import (
    TestTrajectoryOptimizer,
    TestFastPolish,
    TestSolverIterationBudget,
    TestMergeConditions,
    TestBenchmarkDryRun,
    TestSelfConsistentPlan,
    TestSoftWheelForceCost,
    TestPathFitnessScoring,
)
from tests.test_frontier_seed import TestFrontierSeed
from tests.test_unified_planner import TestUnifiedPlanner
from tests.test_candidate_shapes import (
    TestMirrorBow,
    TestSplineGeometry,
    TestReverseSeeding,
    TestCandidateSpecs,
)
from tests.test_robot_model import (
    TestRobotConfig,
    TestDifferentialDriveModel,
    TestMeasuredMotorConfig,
    TestUniversalRobotConfig,
    TestUnifiedMathPrimitives,
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
    TestBadRows,
    TestDistanceScale,
    TestFitWithoutPoses,
    TestIntervalLowerBound,
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
from tests.test_validator_model import (
    TestBacklashModel,
    TestHeadingMetric,
    TestUnifiedReachabilityAndDrivetrainParity,
)
from tests.test_paths import TestCanonicalPaths
from tests.test_polish import (
    TestPlannerCandidateSet,
    TestReachabilityEngine,
    TestGradientPolish,
    TestCandidatePolish,
    TestPlanGates,
)
from tests.test_envelope_soundness import (
    TestAnEmptyTrajectory,
    TestNoPhantomTurnRate,
    TestTheBoxContainsTheReachableSet,
    TestTheTickPropagation,
)
from tests.test_runtime_footprint import TestTheRuntimeFootprint
from tests.test_sysid_compare import TestSysIdCompare
from tests.test_export_compact import TestCompactExport
from tests.test_export import TestResampleHeading
from tests.test_hub_deploy import (
    TestSessionFile,
    TestPushHubCli,
    TestFirmwareCompiles,
    TestHubRunnerContract,
)
from tests.test_hub_ble import (
    TestFindHub,
    TestScan,
    TestHubDeployUsesTheProbe,
    TestTheConnectionFailure,
    TestTheStudioHeadline,
    TestListHubs,
)
from tests.test_sysid_collect import (
    TestCollectorMetadata,
    TestSessionPoses,
    TestFittedScales,
    TestMeasurementSheet,
    TestSheetFormAndOneStepFit,
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
    TestUnifiedDefaults,
    TestFirstRun,
    TestStudioEntryPoints,
    TestDeprecatedCliFlags,
    TestParallelCandidates,
    TestValidationReport,
    TestVariableSampleLayout,
    TestIterationHistory,
    TestCandidateComparison,
)
from tests.test_gui_server import (
    TestStudioRoutes,
    TestStudioSolve,
    TestStudioPush,
    TestStudioHubScan,
    TestStudioModel,
    TestStudioCalibration,
    TestStudioHelpers,
    TestStudioReport,
    TestValidationVerdict,
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
    TestANewSolveClearsTheLastVerdict,
    TestAReloadAttachesToTheRunningSolve,
    TestTheDownloadsBelongToTheirJob,
    TestThePushNeedsAMissionThatMatches,
    TestTheHiddenAttributeWins,
    TestThePushNeedsAValidatedTrajectory,
    TestThePushObeysTheApproval,
    TestAFailedSolveLeavesNoPath,
    TestABestEffortPathIsNotApproved,
    TestASolveThatNeverStartsClearsTheVerdict,
    TestTheMessageTheOperatorReads,
    TestTheResultExplainsItself,
    TestTheSheetWithoutASession,
    TestTheEnvelopeNamesItsModel,
    TestTheCandidateLabelsComeFromTheReport,
)
from tests.test_testcache import (
    TestCacheSkipsUnchangedCode,
    TestCacheFingerprints,
    TestCacheRecords,
)
from tests.test_suite_registry import (
    TestEveryTestClassRuns,
)
from tests.test_research_robots import (
    TestRandomRobots,
)
from tests.test_risk_margins import (
    TestRiskMargins,
)
from tests.test_doc_numbers import (
    TestDocNumbers,
)

# Regression tests (no JAX required)
from tests.test_regressions import (
    TestEventNameEscaping,
    TestGearedSpeedLimit,
    TestSegmentLimits,
    TestExportConfig,
    TestAttemptSelection,
    TestReachabilityEnvelope,
    TestBenchmarkValidationConfig,
    TestExportGate,
    TestValidatorGates,
    TestWriteGateIntegration,
    TestOpenLoopModel,
)

# JAX-dependent tests (skip if jax is not installed)
try:
    from tests.test_jax_ramsete import TestJAXRamsete, TestJAXDriveQuantization
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"Warning: JAX-dependent tests skipped ({e})")
    JAX_AVAILABLE = False


# Every test class in the suite, in the order it runs.
def test_classes():
    """The test classes, the JAX ones first when they are available."""
    classes = [
        TestExportConfig,
        TestSegmentLimits,
        TestGearedSpeedLimit,
        TestEventNameEscaping,
        TestAttemptSelection,
            TestReachabilityEnvelope,
        TestBenchmarkValidationConfig,
        TestExportGate,
        TestValidatorGates,
        TestWriteGateIntegration,
        TestOpenLoopModel,
            TestTrajectoryOptimizer,
        TestRobotConfig,
        TestUniversalRobotConfig,
        TestDifferentialDriveModel,
        TestMeasuredMotorConfig,
        TestUnifiedMathPrimitives,
        TestForwardIntegrate,
        TestAuditConstraints,
        TestComputeMetrics,
        TestValidateTrajectory,
        TestSysIdFit,
        TestBadRows,
        TestDistanceScale,
        TestFitWithoutPoses,
        TestIntervalLowerBound,
        TestRandomSuiteGenerator,
        TestRiskMargins,
        TestDocNumbers,
        TestNoLoadSpeed,
        TestDrivebaseProfile,
        TestHubSessionPoses,
        TestDigitalTwinConfig,
        TestDigitalTwinValidator,
        TestMeasuredMotorFit,
        TestMeasuredMotorEndToEnd,
        TestDigitalTwinMechanisms,
        TestMotorLaw,
        TestDrivetrainModel,
        TestDeclaredMotorCheck,
        TestObstacleOverlap,
        TestFieldCollisions,
        TestFieldFiles,
        TestServerPayload,
        TestLaunchAreas,
        TestRoboVibeField,
        TestRoboVibeConverter,
        TestMatDownload,
        TestAutocrop,
        TestComposeMat,
        TestPdfRendering,
        TestSysIdCompare,
        TestCompactExport,
        TestResampleHeading,
        TestFinalWaypointEvent,
        TestFinalWaypointEventCompact,
        TestFirmwareParity,
        TestLateTickEvents,
        TestSessionFile,
        TestPushHubCli,
        TestFirmwareCompiles,
        TestHubRunnerContract,
        TestFindHub,
        TestScan,
        TestHubDeployUsesTheProbe,
        TestTheConnectionFailure,
        TestTheStudioHeadline,
        TestListHubs,
        TestCollectorMetadata,
        TestSessionPoses,
        TestFittedScales,
        TestMeasurementSheet,
        TestSheetFormAndOneStepFit,
        TestRepeatStatistics,
        TestEveryTestClassRuns,
        TestRefinedStepFit,
        TestDriftRegression,
        TestDriftAndStallTorque,
        TestWallHoming,
        TestCollectCommand,
        TestCliBasics,
        TestUnifiedDefaults,
        TestFirstRun,
        TestStudioEntryPoints,
        TestDeprecatedCliFlags,
        TestStudioPush,
        TestStudioHubScan,
        TestStudioModel,
        TestStudioCalibration,
        TestValidationVerdict,
        TestBacklashModel,
        TestHeadingMetric,
        TestUnifiedReachabilityAndDrivetrainParity,
        TestCanonicalPaths,
        TestPlannerCandidateSet,
        TestReachabilityEngine,
        TestGradientPolish,
        TestCandidatePolish,
        TestPlanGates,
        TestTheRuntimeFootprint,
        TestNoPhantomTurnRate,
        TestTheBoxContainsTheReachableSet,
        TestTheTickPropagation,
        TestAnEmptyTrajectory,
        TestParallelCandidates,
        TestValidationReport,
        TestVariableSampleLayout,
        TestIterationHistory,
        TestCandidateComparison,
        TestFastPolish,
        TestSolverIterationBudget,
        TestMergeConditions,
        TestBenchmarkDryRun,
        TestSelfConsistentPlan,
        TestSoftWheelForceCost,
        TestPathFitnessScoring,
        TestFrontierSeed,
        TestUnifiedPlanner,
        TestMirrorBow,
        TestSplineGeometry,
        TestReverseSeeding,
        TestCandidateSpecs,
        TestStudioRoutes,
        TestStudioSolve,
        TestStudioHelpers,
        TestStudioInventory,
        TestStudioReport,
        TestReportBuilder,
        TestProjectFile,
        TestStudioReportRoute,
        TestStudioInventoryNewControls,
        TestTheHiddenAttributeWins,
        TestANewSolveClearsTheLastVerdict,
        TestAReloadAttachesToTheRunningSolve,
        TestTheDownloadsBelongToTheirJob,
        TestThePushNeedsAMissionThatMatches,
        TestThePushNeedsAValidatedTrajectory,
        TestThePushObeysTheApproval,
        TestAFailedSolveLeavesNoPath,
        TestABestEffortPathIsNotApproved,
        TestASolveThatNeverStartsClearsTheVerdict,
        TestTheMessageTheOperatorReads,
        TestTheResultExplainsItself,
        TestTheSheetWithoutASession,
        TestTheEnvelopeNamesItsModel,
        TestTheCandidateLabelsComeFromTheReport,
        TestCacheSkipsUnchangedCode,
        TestCacheFingerprints,
        TestCacheRecords,
        TestRandomRobots,
    ]
    if JAX_AVAILABLE:
        classes = [
            TestJAXRamsete,
            TestJAXDriveQuantization,
        ] + classes
    return classes


class TimingResult(unittest.TextTestResult):
    """A result that remembers how long every test took."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.durations = []
        self._started = time.perf_counter()

    def startTest(self, test):
        self._started = time.perf_counter()
        super().startTest(test)

    def stopTest(self, test):
        self.durations.append((time.perf_counter() - self._started, test.id()))
        super().stopTest(test)


def parse_flags(argv):
    """The switches the runner understands."""
    flags = {"cache": True, "status": False, "clear": False, "durations": 0}
    for index, arg in enumerate(argv):
        if arg == "--no-cache":
            flags["cache"] = False
        elif arg == "--cache-status":
            flags["status"] = True
        elif arg == "--clear-cache":
            flags["clear"] = True
        elif arg == "--durations":
            if index + 1 < len(argv):
                flags["durations"] = int(argv[index + 1])
        elif arg.startswith("--durations="):
            flags["durations"] = int(arg.split("=", 1)[1])
    return flags


def test_ids(classes):
    """The id of every test in the suite."""
    ids = []
    for klass in classes:
        for name in dir(klass):
            if name.startswith("test") and callable(getattr(klass, name)):
                ids.append(f"{klass.__module__}.{klass.__qualname__}.{name}")
    return ids


def print_cache_status(cache, classes):
    """Say what the cache would do, without running a thing."""
    changed = cache.changed_files()
    ids = test_ids(classes)
    skipped = sum(1 for test_id in ids if cache.should_skip(test_id))
    print("TEST CACHE")
    print(f"  the cache file: {cache.path}")
    print(f"  {len(cache.entries)} tests have a fingerprint, {len(ids)} in the suite")
    print(f"  {skipped} of them would be skipped, {len(ids) - skipped} would run")
    if changed:
        print(f"  {len(changed)} file(s) changed since the last run:")
        for rel in changed[:40]:
            print(f"    {rel}")
        if len(changed) > 40:
            print(f"    ... and {len(changed) - 40} more")
    else:
        print("  nothing changed since the last run")
    print(f"  environment: {cache.env()}")


def run_tests():
    """Run all tests and return results."""
    flags = parse_flags(sys.argv[1:])
    classes = test_classes()

    if flags["clear"]:
        if testcache.CACHE_PATH.exists():
            testcache.CACHE_PATH.unlink()
            print("The test cache is gone; every test runs again.")
        else:
            print("There was no test cache.")

    cache = testcache.Cache(enabled=flags["cache"])

    if flags["status"]:
        print_cache_status(cache, classes)
        return True

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for klass in classes:
        suite.addTests(loader.loadTestsFromTestCase(klass))

    # The cache decides which tests do not have to run again.
    testcache.instrument(classes, cache)

    # Run tests
    started = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2, resultclass=TimingResult)
    result = runner.run(suite)
    elapsed = time.perf_counter() - started

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    if cache.skipped:
        print(f"  of which {len(cache.skipped)} by the cache: "
              f"nothing they read has changed")
    print(f"Time: {elapsed:.1f} s")
    if cache.saved_seconds:
        print(f"The cache saved {cache.saved_seconds:.1f} s of runtime")

    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nError tests:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    # The one way to exit 1 with no failure and no error: a test marked
    # expectedFailure that passed. It used to be silent.
    if result.unexpectedSuccesses:
        print("\nUnexpected successes (marked expectedFailure but passed):")
        for test in result.unexpectedSuccesses:
            print(f"  - {test}")

    if flags["durations"] and result.durations:
        print("\nSlowest tests:")
        for duration, test_id in sorted(result.durations, reverse=True)[:flags["durations"]]:
            print(f"  {duration:7.2f} s  {test_id}")

    cache.save()

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
