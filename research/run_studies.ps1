# The measurement suite for the research agenda.
#
# Everything runs against a PINNED copy of the planner, because the working
# tree is edited while a measurement runs (KI-024 landed mid-session once).
# The pinned tree is recorded in docs/research/pq_b_PINNED_TREE.md5.
#
#   $env:OVERBOARD_TREE = <pinned copy>
#   powershell -File research/run_studies.ps1
#
# The runs:
#
#   matrix.jsonl      the seed / timing / gate matrix on the whole suite
#   weights.jsonl     Q6  - is the accuracy weight a dial or a lottery?
#   density.jsonl     Q7  - sample density against timing flexibility
#   multiverse.jsonl  Q12 - does the experimental pipeline ever win?
#   spread.jsonl      KI-020 - how much of a PASS depends on the declared
#                            uncertainty ranges?

$ErrorActionPreference = 'Stop'

if (-not $env:OVERBOARD_TREE) {
    Write-Output 'OVERBOARD_TREE is not set; measuring the working tree.'
    Write-Output 'Set it to a pinned copy for a reproducible measurement.'
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
$random = 0..23 | ForEach-Object { "research/missions/m{0:D3}.json" -f $_ }

# The weight ladder and the uncertainty sensitivity run on 12 missions.
$study = $examples[0..5] + $random[0..5]
# The density study runs on 8, because a 40-sample-per-segment solve on a
# 7-segment mission is expensive.
$density = $examples[0..3] + $random[0..3]

New-Item -ItemType Directory -Force -Path research/results | Out-Null

Write-Output '=== the main matrix ==='
python research/harness.py --missions $examples $random `
    --out research/results/matrix.jsonl --specs matrix --samples 10 --workers 8

Write-Output '=== Q6: wide accuracy-weight ladder ==='
python research/harness.py --missions $study --out research/results/weights.jsonl `
    --specs wide-weights --samples 10 --workers 8 --no-critic

Write-Output '=== Q7: sample density ==='
python research/harness.py --missions $density --out research/results/density.jsonl `
    --specs density --workers 8 --no-critic

Write-Output '=== Q12: the Multi-Verse pipeline ==='
python research/harness.py --missions $study --out research/results/multiverse.jsonl `
    --specs multiverse --samples 10 --workers 8 --no-critic

Write-Output '=== KI-020: uncertainty sensitivity ==='
python research/harness.py --missions $study --out research/results/spread.jsonl `
    --specs seeds --samples 10 --workers 8 --no-critic --spreads 0 0.5 1 1.5 2

Write-Output '=== done ==='
