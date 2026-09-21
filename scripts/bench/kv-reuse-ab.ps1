# R2T2 cross-tick KV continuation: does it change the transcript, and what does
# it buy?
#
# The reuse is the one streaming cache in this repo whose exactness is a
# *mathematical* argument (causality) rather than bit-equality: the rows it
# keeps were computed inside a differently-shaped graph, so they can differ from
# a from-scratch prefill in the last bits. The claim that has to hold is
# therefore the same as for the other two caches -- identical transcript -- and
# it is checked, not argued: at every cadence, reuse-on and reuse-off must agree
# with each other.
#
# Only the streaming arms can differ: `kv_reuse` is hardwired false everywhere
# else (the only `true` is the r2t2 stream's decode call), so the offline
# single-pass run is an independent witness of the audio rather than a member of
# the comparison set. It is expected to differ by a character or two -- a
# one-shot decode and a tick-by-tick decode do not segment or punctuate
# identically (the offline run says "forward," where streaming says "forward;")
# -- and that difference predates any cache here.
#
# The reuse also has to be shown to have *engaged*, or an agreement is vacuous
# and the timings are a no-op. Both arms are therefore checked against the
# prefilled-position count run_decode_pass reports per tick.
#
# Measured through a file, not through the merged terminal stream. The first
# version of this script redirected `2>&1` and compared the `text:` line, which
# cost an afternoon: the transcript goes to **stdout** (a printf in the CLI) and
# every `[info]`/`[debug]` line goes to **stderr** (log_msg), so fusing them
# lets an unbuffered stderr line land inside the block-buffered stdout text.
# Two runs came back 213 and 209 chars with distinct hashes -- looking exactly
# like a cache that changes the transcript -- when the file actually held the
# full 586 chars with a timings line spliced into the middle of it. So now:
# stdout and stderr go to separate files, the transcript is taken from the
# file the CLI writes itself (`-o`, one writer, nothing to interleave with),
# and the stream's own `finalize ... text=N bytes committed=M` accounting is
# carried as a second, independent witness.
#
# Protocol: arms interleaved (kv, nokv, kv, nokv, ...) so thermal/background
# drift hits both equally, 4 runs per arm at 80 ms (first dropped, rest
# averaged) and 2 at 160/320.
param(
    [string]$Wav = 'samples\dots.wav',
    [int]$Reps = 4,
    # Cadences to measure. The full matrix is 80/160/320; a confirmation run
    # after a change that cannot move the numbers re-measures 80 only.
    [int[]]$Cadences = @(80, 160, 320)
)
$ErrorActionPreference = 'Stop'
Set-Location 'C:\Users\Z\Downloads\PROJECTS\transcribe-fork'

$model = "$env:USERPROFILE\.cache\huggingface\hub\models--davidxifeng--Confucius4-R2T2-gguf\snapshots\a8e6b385d7df7eae9519363e07034a209004797a\r2t2-q8_0.gguf"
$exe   = "build\maxx-cuda-graphs\bin\transcribe-cli.exe"

function Hash([string]$s) {
    if ([string]::IsNullOrEmpty($s)) { return '(empty)' }
    [System.BitConverter]::ToString([System.Security.Cryptography.SHA1]::Create().ComputeHash(
        [System.Text.Encoding]::UTF8.GetBytes($s))).Substring(0, 11)
}
function Series($lines, $re) { @($lines | ForEach-Object { if ($_ -match $re) { [double]$Matches[1] } }) }
function Stat($v) {
    if ($v.Count -eq 0) { return '(none)' }
    "first={0,6:N1} mean={1,6:N1} max={2,6:N1} last={3,6:N1}" -f `
        $v[0], ($v | Measure-Object -Average).Average, ($v | Measure-Object -Maximum).Maximum, $v[-1]
}

# One run. Returns everything the report needs, so the report never re-parses.
function Run-Arm([int]$ms, [string]$arm, [int]$rep) {
    $tag = if ($ms -gt 0) { "$arm-$ms-$rep" } else { "$arm-offline" }
    $out = Join-Path $env:TEMP "kv-$tag.out"   # stdout: the CLIs own summary
    $err = Join-Path $env:TEMP "kv-$tag.err"   # stderr: every log + perf line
    $txt = Join-Path $env:TEMP "kv-$tag.txt"   # -o: the transcript, single writer
    $pre = if ($arm -eq 'nokv') { 'set TRANSCRIBE_R2T2_NO_KV_REUSE=1&& ' } else { '' }
    $arg = if ($ms -gt 0) { "--stream-chunk-ms $ms" } else { "" }

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    cmd /c "$pre`set TRANSCRIBE_PERF_DEBUG=1&& $exe -m `"$model`" $Wav --backend cuda $arg -o `"$txt`" > `"$out`" 2> `"$err`""
    $code = $LASTEXITCODE
    $sw.Stop()

    $lines = @(Get-Content $err -ErrorAction SilentlyContinue)
    # The transcript, from the file the CLI wrote. Trailing newlines are
    # normalized (write_output_file appends one) -- the byte count the stream
    # reports is carried separately below, so nothing is hidden by the trim.
    $text = if (Test-Path $txt) { ([System.IO.File]::ReadAllText($txt)).TrimEnd("`r", "`n") } else { '' }
    # Second witness: the stream's own accounting, independent of any printing.
    $fin = [regex]::Match(($lines -join "`n"),
        'finalize ticks=(\d+) audio=([\d.]+)s text=(\d+) bytes committed=(\d+)')
    # Per-tick prefill geometry: "prefill_compute  75.24 ms  (T_prompt=591, 470
    # from the KV cache, 121 prefilled)". reused/rows is the fraction of the
    # prompt the pass did not recompute.
    $reused = Series $lines 'prefill_compute\s+[0-9.]+ ms\s+\(T_prompt=(\d+), (\d+) from'
    $tprom  = Series $lines 'prefill_compute\s+[0-9.]+ ms\s+\(T_prompt=(\d+),'
    $graph  = Series $lines '\(T_prompt=\d+, \d+ from the KV cache, (\d+) prefilled\)'
    [pscustomobject]@{
        arm     = $arm
        ms      = $ms
        rep     = $rep
        exit    = $code
        chars   = $text.Length
        sha     = (Hash $text)
        text    = $text
        # finalize witness (empty for the offline run, which has no stream)
        ftick   = if ($fin.Success) { [int]$fin.Groups[1].Value } else { -1 }
        ftext   = if ($fin.Success) { [int]$fin.Groups[3].Value } else { -1 }
        fcomm   = if ($fin.Success) { [int]$fin.Groups[4].Value } else { -1 }
        wall    = $sw.Elapsed.TotalMilliseconds
        mel     = Series $lines '^\s*mel\s+([0-9.]+) ms'
        enc     = Series $lines 'enc_compute\s+([0-9.]+) ms'
        pre     = Series $lines 'prefill_compute\s+([0-9.]+) ms'
        pbuild  = Series $lines 'prefill_build\s+([0-9.]+) ms'
        steps   = Series $lines 'step_loop\s+([0-9.]+) ms'
        tick    = @(Series $lines 'tick_us=(\d+)' | ForEach-Object { $_ / 1000.0 })
        tprom   = $tprom
        graph   = $graph
        # The signature of an engaged reuse: how many ticks prefilled fewer rows
        # than the prompt holds, and the mean prefilled fraction.
        engaged = @($graph | Where-Object { $_ -gt 0 }).Count
        ticks   = $graph.Count
    }
}

Write-Output "model: $([System.IO.Path]::GetFileName($model))"
Write-Output "wav:   $Wav"

$rows = @()
# Offline reference first: the transcript the streaming arms are held against.
# One run, because it is a reference value and not a timing arm -- it pins what
# the streaming arms are compared *to*, and its own wall time is not a
# measurement. Skipped when the caller asks for a single cadence.
if ($Cadences.Count -gt 1) { $rows += Run-Arm 0 'kv' 0 }

foreach ($ms in $Cadences) {
    $reps = if ($ms -eq 80) { $Reps } else { 2 }
    for ($r = 1; $r -le $reps; $r++) {
        foreach ($arm in @('kv', 'nokv')) {
            $rows += Run-Arm $ms $arm $r
        }
    }
}

$stream = @($rows | Where-Object { $_.ms -gt 0 })

Write-Output "`n=== transcript (chars / sha1 of the text the CLI wrote) ==="
foreach ($row in $rows) {
    $label = if ($row.ms -gt 0) { "$($row.arm) $($row.ms)ms r$($row.rep)" } else { "$($row.arm) offline" }
    $w = if ($row.fcomm -ge 0) { "stream: $($row.ftick) ticks, $($row.fcomm) bytes committed" } else { '(no stream)' }
    Write-Output ("{0,-16} exit={1} chars={2,4} sha={3} wall={4,7:N0} ms   {5}" -f `
        $label, $row.exit, $row.chars, $row.sha, $row.wall, $w)
}

# Correctness: every streaming arm/rep/cadence must agree, and the stream's own
# accounting must agree with the text it wrote.
$shas = @($stream | ForEach-Object { $_.sha } | Select-Object -Unique)
$comm = @($stream | ForEach-Object { $_.fcomm } | Select-Object -Unique)
$mismatch = @($stream | Where-Object { $_.fcomm -ge 0 -and $_.fcomm -ne $_.chars })
Write-Output ""
if ($shas.Count -eq 1 -and $comm.Count -eq 1 -and $mismatch.Count -eq 0) {
    Write-Output "VERDICT: PASS - all $($stream.Count) streaming runs agree ($($shas[0]), $($stream[0].chars) chars, $($comm[0]) bytes committed)"
    $off = @($rows | Where-Object { $_.ms -eq 0 })
    if ($off.Count -gt 0) {
        Write-Output "  offline witness: $($off[0].chars) chars, sha $($off[0].sha)$(if ($off[0].sha -ne $shas[0]) { ' (differs, as a one-shot decode does)' })"
    }
} else {
    Write-Output "VERDICT: FAIL - streaming transcripts disagree"
    foreach ($sha in $shas) {
        $hit = @($stream | Where-Object { $_.sha -eq $sha })
        Write-Output "  $sha <- $(($hit | ForEach-Object { "$($_.arm)/$($_.ms)ms r$($_.rep)" }) -join ' ')"
    }
    foreach ($row in $mismatch) {
        Write-Output "  $($row.arm)/$($row.ms)ms r$($row.rep): wrote $($row.chars) chars but stream reports $($row.fcomm) committed"
    }
}

Write-Output "`n=== reuse engagement (ticks that prefilled less than the prompt) ==="
foreach ($row in $stream) {
    $label = "$($row.arm) $($row.ms)ms r$($row.rep)"
    if ($row.ticks -eq 0) { Write-Output "$label : no perf lines"; continue }
    $frac = if ($row.tprom.Count -gt 0) {
        ($row.graph | Measure-Object -Average).Average / ($row.tprom | Measure-Object -Average).Average
    } else { 0 }
    Write-Output ("{0,-16} engaged {1,3}/{2,3} ticks   prefilled mean={3,6:N1} of T_prompt mean={4,6:N1} ({5,5:P0} recomputed)" -f `
        $label, $row.engaged, $row.ticks, ($row.graph | Measure-Object -Average).Average,
        ($row.tprom | Measure-Object -Average).Average, $frac)
}

Write-Output "`n=== per-tick stage cost, 80 ms, first rep dropped ==="
Write-Output ("{0,-8} {1,-14} {2,-42}" -f 'arm', 'stage', 'first / mean / max / last (ms)')
foreach ($arm in @('kv', 'nokv')) {
    $sel = @($rows | Where-Object { $_.ms -eq 80 -and $_.arm -eq $arm -and $_.rep -gt 1 })
    foreach ($stage in @('mel', 'enc', 'pre', 'pbuild', 'steps', 'tick')) {
        $v = @($sel | ForEach-Object { $_.$stage } | Where-Object { $null -ne $_ })
        if ($v.Count -eq 0) { continue }
        Write-Output ("{0,-8} {1,-14} {2}" -f $arm, $stage, (Stat $v))
    }
}

Write-Output "`n=== wall clock, 80 ms (real-time factor; first rep dropped) ==="
foreach ($arm in @('kv', 'nokv')) {
    $sel = @($rows | Where-Object { $_.ms -eq 80 -and $_.arm -eq $arm -and $_.rep -gt 1 })
    if ($sel.Count -eq 0) { continue }
    $wall = ($sel | Measure-Object -Property wall -Average).Average
    Write-Output ("{0,-8} {1,7:N0} ms  ({2:N2}x realtime, {3} runs)" -f $arm, $wall, (35332.0 / $wall), $sel.Count)
}
