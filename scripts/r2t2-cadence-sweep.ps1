# scripts/r2t2-cadence-sweep.ps1
#
# Cadence sweep for the Confucius4-R2T2 streaming path: feeds the same file at
# every chunk size the streaming API is expected to accept, so the low end can
# be pushed as far down as the model tolerates ("lowest latency possible").
#
# Why a sweep and not one number: R2T2 re-encodes the whole accumulated audio
# on every tick, so a smaller chunk does not just shorten the latency -- it
# multiplies the number of ticks and therefore the total compute. The cost
# curve is the thing being measured, alongside the two correctness gates that
# must hold at every cadence:
#   - the final transcript equals the offline transcript, and
#   - committed text only ever grows (the append-only guarantee), which the CLI
#     reports via a non-zero exit and a COMMITTED-TEXT VIOLATION line.
#
# --stream-chunk-ms cannot be combined with --repeat, so the repeats here are
# separate processes. Per the repo's benchmark rule the first run of each arm
# is discarded as cold (lazy init dominates it) and the rest are averaged.

param(
    [string]   $Model      = "$env:USERPROFILE\.cache\huggingface\hub\models--davidxifeng--Confucius4-R2T2-gguf\snapshots\a8e6b385d7df7eae9519363e07034a209004797a\r2t2-q8_0.gguf",
    [string]   $Wav        = "samples\jfk.wav",
    [string]   $Backend    = "cuda",
    [int[]]    $Cadences   = @(80, 160, 320, 640, 1280, 2000),
    [int]      $Runs       = 3,
    [string]   $Exe        = "build\maxx-cuda-graphs\bin\transcribe-cli.exe",
    [string]   $Reference  = ""
)

$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)

# The reference text every cadence must reproduce. Default: derive it from the
# first cadence that succeeds, so the sweep is self-checking rather than
# depending on a transcript pasted into this script.
$refText = if ($Reference) { $Reference } else { $null }

$noise = 'CUDA Graph|graph_warmup|ggml_cuda_init|Device 0|load_backend|dynamic architecture|R2T2 package|converted|using cuda|^$'

$results = @()
foreach ($ms in $Cadences) {
    # NOT `$runs`: PowerShell variable names are case-insensitive, so that
    # would alias the [int]$Runs parameter and fail on array assignment.
    $armRuns = @()
    for ($i = 1; $i -le $Runs; $i++) {
        $log = Join-Path $env:TEMP "r2t2-sweep-$ms-$i.log"
        $sw  = [System.Diagnostics.Stopwatch]::StartNew()
        & $Exe -m $Model $Wav --backend $Backend --stream-chunk-ms $ms --quiet > $log 2>&1
        $code = $LASTEXITCODE
        $sw.Stop()
        $lines = Get-Content $log | Where-Object { $_ -notmatch $noise }

        $text = ($lines | Where-Object { $_ -match '^text: ' } | Select-Object -First 1) -replace '^text: ', ''
        $lang = ($lines | Where-Object { $_ -match '^detected-language: ' } | Select-Object -First 1) -replace '^detected-language: ', ''
        # The stream-wall line is the CLI's own timing of begin->finalize. NOT
        # the `realtime:` line: that one is computed from
        # transcribe_get_timings, which for a stream only sees the last tick
        # and reports a ~100 ms figure that has nothing to do with streaming
        # cost. Parsing it here would make every cadence look identical.
        $wallLine = ($lines | Where-Object { $_ -match 'stream-wall:' } | Select-Object -First 1)
        $wallMs   = if ($wallLine -and $wallLine -match 'stream-wall:\s+([0-9.]+) ms') { [double]$Matches[1] } else { 0 }
        $wallX    = if ($wallLine -and $wallLine -match '\(([0-9.]+)x realtime\)') { [double]$Matches[1] } else { 0 }
        $feedLine = ($lines | Where-Object { $_ -match 'stream-feeds:' } | Select-Object -First 1)
        $feedMax  = if ($feedLine -and $feedLine -match 'max=([0-9.]+) ms') { [double]$Matches[1] } else { 0 }
        $headLine = ($lines | Where-Object { $_ -match 'stream-headroom:' } | Select-Object -First 1)
        $headroom = if ($headLine -and $headLine -match 'is ([0-9.]+)x') { [double]$Matches[1] } else { 0 }
        $violation = [bool]($lines | Where-Object { $_ -match 'VIOLATION' })
        $feeds = @($lines | Where-Object { $_ -match '^\s+feed\[' }).Count
        $tickUs = @($lines | Where-Object { $_ -match 'tick_us=(\d+)' } | ForEach-Object {
            if ($_ -match 'tick_us=(\d+)') { [int]$Matches[1] } })
        $lastTickUs = if ($tickUs.Count -gt 0) { $tickUs[-1] } else { 0 }
        $meanTickUs = if ($tickUs.Count -gt 0) { [int](($tickUs | Measure-Object -Average).Average) } else { 0 }
        $finalize = ($lines | Where-Object { $_ -match 'finalize: ticks=' } | Select-Object -First 1)

        $armRuns += [pscustomobject]@{
            Run = $i; Exit = $code; Text = $text; Lang = $lang
            WallMs = $wallMs; WallX = $wallX; FeedMaxMs = $feedMax
            Headroom = $headroom; Feeds = $feeds
            MeanTickUs = $meanTickUs; LastTickUs = $lastTickUs
            Violation = $violation; Finalize = $finalize
        }
    }

    $warm = $armRuns | Select-Object -Skip 1
    if (-not $warm) { $warm = $armRuns }   # Runs=1: nothing to discard
    $ok   = ($armRuns | Where-Object { $_.Exit -ne 0 -or $_.Violation }).Count -eq 0
    $text0 = $armRuns[0].Text
    $sameText = ($armRuns | Where-Object { $_.Text -ne $text0 }).Count -eq 0
    if (-not $refText) { $refText = $text0 }
    $matchesRef = ($text0 -eq $refText)

    $meanWall = [math]::Round((($warm | Measure-Object WallMs -Average).Average), 1)
    $results += [pscustomobject]@{
        CadenceMs  = $ms
        WarmWallMs = $meanWall
        # Against the same 11 s reference audio; derived from the mean wall so
        # the two columns can never disagree.
        WarmX      = if ($meanWall -gt 0) { [math]::Round(11000.0 / $meanWall, 1) } else { 0 }
        Feeds      = $armRuns[0].Feeds
        FeedMaxMs  = [math]::Round((($warm | Measure-Object FeedMaxMs -Average).Average), 1)
        Headroom   = [math]::Round($armRuns[0].Headroom, 2)
        MeanTickUs = $armRuns[0].MeanTickUs
        LastTickUs = $armRuns[0].LastTickUs
        Lang       = $armRuns[0].Lang
        AllRunsOk  = $ok
        StableText = $sameText
        MatchesRef = $matchesRef
        RefText    = $text0
    }
}

""
"=== R2T2 cadence sweep ($Backend, $Runs runs/arm, first discarded as cold) ==="
$results | Format-Table CadenceMs, WarmWallMs, WarmX, Feeds, FeedMaxMs, Headroom, MeanTickUs, LastTickUs, Lang, AllRunsOk, StableText, MatchesRef -AutoSize
""
"reference text: $refText"
""
"reading the table:"
"  Headroom   = slowest single feed / cadence. <= 1.0 means the model keeps"
"               up with real time at that chunk size; > 1.0 means it cannot"
"               sustain live audio there and will fall behind."
"  LastTickUs vs MeanTickUs = how far per-tick cost has grown by the end of"
"               an 11 s utterance. R2T2 re-encodes all accumulated audio"
"               each tick, so this ratio is the direct cost of that design"
"               and it, not the cadence itself, sets the usable floor."
"  MatchesRef = the streamed transcript equals the shortest-cadence one."
"  AllRunsOk  = no non-zero exit and no COMMITTED-TEXT VIOLATION in any run."
