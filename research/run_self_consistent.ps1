# RQ-A's OQ6: do the seed conclusions survive the discretisation fix?
#
# KI-026 added --self-consistent: the samples lie on the path the plan's own
# (v, omega) profile traces. Every plan changes, so the candidate set and the
# seed conclusions have to be re-measured under the flag.
#
# Both modes run on the SAME pinned tree, so the flag is the only difference.
#
#   $env:OVERBOARD_TREE = "<pinned copy>"
#   powershell -File research/run_self_consistent.ps1

$ErrorActionPreference = 'Stop'

if (-not $env:OVERBOARD_TREE) {
    Write-Output 'OVERBOARD_TREE is not set; measuring the working tree.'
}

$examples = @(
    'examples/example_straight.json',
    'examples/example_s_curve.json',
    'examples/example_mission_first4.json',
    'examples/example_complex_mission.json',
    'examples/example_complex_segment.json',
    'examples/example_with_stops.json',
    'examples/example_with_events.json',
    'examples/example_complete.json'
)
$random = 0..7 | ForEach-Object { "research/missions/m{0:D3}.json" -f $_ }
$missions = $examples + $random

New-Item -ItemType Directory -Force -Path research/results | Out-Null

Write-Output '=== the same tree, trapezoidal collocation (the default) ==='
python research/harness.py --missions $missions `
    --out research/results/sc_off.jsonl --specs seeds --samples 10 --workers 8

Write-Output '=== the same tree, self-consistent collocation (KI-026) ==='
python research/harness.py --missions $missions `
    --out research/results/sc_on.jsonl --specs seeds --samples 10 --workers 8 `
    --self-consistent

Write-Output '=== done ==='
