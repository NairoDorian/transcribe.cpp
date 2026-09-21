"""Paired, interleaved A/B of two native libraries on one case.

suite.py answers "how fast is this tree?"; this answers "is arm B slower than
arm A, beyond noise?". Those need different designs. A suite runs one arm's
cases back to back, so machine drift (thermal, allocator state, a background
build) lands entirely inside the arm that happened to run second. Here the arms
alternate within each repetition and the starting arm rotates, so drift is
shared between them.

Each arm loads through the Python bindings of the checkout that owns its
library. The bindings are generated against that checkout's ABI, so a foreign
library fails inside the generated module at import time; that is a hard error,
never something to silently time.

The verdict is deliberately conservative. A difference is only called a
regression when it exceeds the larger arm's own within-arm spread, which is the
noise floor this case actually exhibits rather than a guess. Anything smaller is
reported as within-noise and must not be used to justify a change.

That floor is unpaired, so on a case long enough to drift it goes wide enough to
hide a small real effect. The interleaving already produces matched pairs — the
arms are measured back to back within each repetition — so the pair differences
are reported as well, with a sign test. Treat them as the more sensitive read
and the verdict above as the decision rule; at the default three repetitions an
all-wins pair can only reach p=0.25, so use --reps 5 or more to resolve a small
effect this way.

An arm may carry its own environment (`--arm-env NAME=KEY=VALUE`). Several
optimizations here are gated at run time rather than build time, so this is what
lets them be compared without rebuilding between arms, and the gate each arm ran
under is recorded in the JSON.

Usage:
    uv run --no-project scripts/bench/compare.py \
        --wav samples/jfk.wav --model <installed.gguf> --backend cpu \
        --arm ref=../transcribe_benchmarks/build/bench-native/install/bin/transcribe.dll \
        --arm fork=build/bench-native/install/bin/transcribe.dll

    # Runtime-gated feature: same library, one arm with the pass enabled.
    uv run --no-project scripts/bench/compare.py \
        --wav samples/jfk.wav --model <installed.gguf> --backend cuda \
        --arm off=build/bench-native/install/bin/transcribe.dll \
        --arm on=build/bench-native/install/bin/transcribe.dll \
        --arm-env on=TRANSCRIBE_GRAPH_OPTIMIZER=1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


def tree_bindings(library):
    """Bindings of the checkout that owns `library`, or None when not found.

    A library under `<tree>/build/.../transcribe.dll` belongs to `<tree>`, and
    `<tree>/bindings/python/src` is generated for its ABI. Loading another
    tree's library through them fails on the first symbol it does not export.
    """
    for parent in library.parents:
        candidate = parent / "bindings" / "python" / "src"
        if (candidate / "transcribe_cpp" / "__init__.py").is_file():
            return candidate
    return None


def run_arm(name, library, bindings, args, workdir, extra_env=None):
    """One pipeline.py invocation: 1 warm-up + 2 measured runs, mean returned.

    The child's output is kept rather than discarded: when an arm cannot load,
    the reason is in there, and a swallowed traceback is indistinguishable from
    an unexplained crash.

    `extra_env` carries the arm's runtime feature toggles. Several optimizations
    in this tree are gated by environment variable at run time rather than at
    build time (TRANSCRIBE_GRAPH_OPTIMIZER, GGML_CUDA_GRAPH_OPT,
    GGML_CUDA_DISABLE_GRAPHS), so a per-arm environment is what lets them be
    compared without rebuilding the library between arms.
    """
    serial = run_arm.serial
    run_arm.serial += 1
    output = workdir / f"{name}.{serial}.json"
    log = workdir / f"{name}.{serial}.log"
    cmd = ["uv", "run", "--no-project", str(Path(__file__).with_name("pipeline.py")),
           "--model", str(args.model), "--wav", str(args.wav),
           "--library", str(library), "--backend", args.backend,
           "--language", args.language, "--threads", str(args.threads),
           "--output", str(output)]
    if args.stream_chunk_ms:
        cmd += ["--stream-chunk-ms", str(args.stream_chunk_ms)]
    if args.att_right is not None:
        cmd += ["--att-right", str(args.att_right)]
    cmd += ["--bindings", str(bindings)]
    env = dict(os.environ, **extra_env) if extra_env else None
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(cmd, stdout=handle, stderr=handle, env=env).returncode
    if code:
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
        raise SystemExit(f"arm {name} failed (exit {code}); last lines of {log}:\n{tail}")
    report = json.loads(output.read_text(encoding="utf-8"))
    return report["warm_mean_ms"], report["runs"][-1]["text"]


run_arm.serial = 0


def median_and_spread(samples):
    ordered = sorted(samples)
    return statistics.median(ordered), ordered[-1] - ordered[0]


def sign_test(wins, trials):
    """Exact two-sided sign-test p-value for `wins` of `trials` decided pairs.

    Ties are excluded, since a tied pair says nothing about direction. At the
    default three repetitions the best attainable value is 0.25, so this can
    only become evidence at five pairs or more; that ceiling is stated in the
    output rather than hidden, because a p-value read at the wrong n is worse
    than no p-value.
    """
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(wins, trials + 1))
    return min(1.0, 2 * tail / 2 ** trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--arm", required=True, action="append", metavar="NAME=LIBRARY",
                        help="repeatable; the first arm is the baseline")
    parser.add_argument("--arm-bindings", action="append", metavar="NAME=PATH",
                        help="override the auto-detected bindings for one arm")
    parser.add_argument("--arm-env", action="append", metavar="NAME=KEY=VALUE",
                        help="environment variable to set for one arm only (repeatable); "
                             "use for runtime feature gates such as TRANSCRIBE_GRAPH_OPTIMIZER=1")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--backend", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--language", default="auto")
    parser.add_argument("--stream-chunk-ms", type=int, default=0)
    parser.add_argument("--att-right", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    overrides = {}
    for spec in args.arm_bindings or []:
        name, _, path = spec.partition("=")
        if not name or not path:
            parser.error(f"--arm-bindings must be NAME=PATH, got {spec!r}")
        overrides[name] = Path(path).resolve()
    env_overrides = {}
    for spec in args.arm_env or []:
        name, _, assignment = spec.partition("=")
        key, sep, value = assignment.partition("=")
        if not name or not key or not sep:
            parser.error(f"--arm-env must be NAME=KEY=VALUE, got {spec!r}")
        env_overrides.setdefault(name, {})[key] = value
    arms = []
    for spec in args.arm:
        name, _, library = spec.partition("=")
        if not name or not library:
            parser.error(f"--arm must be NAME=LIBRARY, got {spec!r}")
        path = Path(library)
        if not path.is_file():
            parser.error(f"library not found: {path}")
        bindings = overrides.get(name) or tree_bindings(path.resolve())
        if bindings is None:
            parser.error(f"no bindings found for arm {name}; "
                         f"pass --arm-bindings {name}=<checkout>/bindings/python/src")
        arms.append((name, path.resolve(), bindings, env_overrides.get(name)))
    unknown_env = set(env_overrides) - {name for name, _, _, _ in arms}
    if unknown_env:
        parser.error(f"--arm-env names no such arm: {', '.join(sorted(unknown_env))}")
    if args.reps < 2:
        parser.error("--reps must be at least 2; with one repetition there is no noise estimate")

    samples = {name: [] for name, _, _, _ in arms}
    texts = {name: set() for name, _, _, _ in arms}
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for rep in range(args.reps):
            # Rotate the starting arm so no arm is always measured first.
            for offset in range(len(arms)):
                name, library, bindings, extra_env = arms[(rep + offset) % len(arms)]
                value, text = run_arm(name, library, bindings, args, workdir, extra_env)
                samples[name].append(value)
                texts[name].add(text)
                print(f"rep {rep + 1}/{args.reps} {name}: {value:.1f} ms", file=sys.stderr, flush=True)

    stats = {name: median_and_spread(values) for name, values in samples.items()}
    baseline_name = arms[0][0]
    baseline_median, baseline_spread = stats[baseline_name]
    # Arms are 4-tuples; keep the names separate so the reporting below does not
    # have to unpack a tuple it does not otherwise need.
    arm_names = [name for name, _, _, _ in arms]
    result = {"case": str(args.model.name), "backend": args.backend, "reps": args.reps,
              "policy": "interleaved arms, rotating start; per arm 1 warm-up + 2 measured runs",
              "arms": {name: {"median_ms": stats[name][0], "spread_ms": stats[name][1],
                              "samples_ms": samples[name],
                              "env": env_overrides.get(name) or {},
                              "transcripts": len(texts[name])} for name in arm_names}}
    if env_overrides:
        # Recorded because a timing is meaningless without the gates it ran under.
        result["arm_env"] = env_overrides

    print(f"\ncase: {args.model.name}  backend={args.backend}  reps={args.reps}")
    print(f"{'arm':<12}{'median ms':>12}{'spread ms':>12}{'vs base':>10}{'transcripts':>13}")
    for name in arm_names:
        median, spread = stats[name]
        ratio = median / baseline_median
        result["arms"][name]["ratio_to_baseline"] = ratio
        print(f"{name:<12}{median:>12.1f}{spread:>12.1f}{ratio:>10.3f}{len(texts[name]):>13}")

    parity = {name: len(texts[name]) for name in arm_names}
    result["transcript_parity"] = parity
    differing = [name for name, count in parity.items() if count > 1]
    if differing:
        result["transcript_mismatch"] = differing
        print(f"\nWARNING: {', '.join(differing)} produced more than one transcript; "
              "timings for that arm are not comparable")
    elif len({next(iter(texts[name])) for name in arm_names}) > 1:
        result["arms_disagree"] = True
        print("\nNOTE: arms produced different transcripts; the timings are NOT a "
              "like-for-like comparison. Inspect the text before reading anything into these numbers.")

    print()
    for name in arm_names[1:]:
        median, spread = stats[name]
        ratio = median / baseline_median
        noise = max(spread, baseline_spread) / baseline_median
        if ratio > 1 + noise:
            verdict = f"SLOWER than {baseline_name} beyond noise ({noise * 100:.1f}%)"
        elif ratio < 1 - noise:
            verdict = f"faster than {baseline_name} beyond noise ({noise * 100:.1f}%)"
        else:
            verdict = f"within noise ({noise * 100:.1f}%)"
        result["arms"][name]["verdict"] = verdict
        print(f"{name} vs {baseline_name}: {ratio:.3f}  -> {verdict}")

    # Paired read. The arms alternate within each repetition and the starting
    # arm rotates, so repetition r is a matched pair: both arms were measured
    # back to back under near-identical thermal and allocator state. Preserving
    # that pairing is the entire reason the arms are interleaved, and reading it
    # directly cancels the drift that otherwise has to be absorbed by the
    # unpaired floor above — which is what leaves that floor too wide to resolve
    # a small effect. Reported alongside, never instead of, the verdict.
    paired = {}
    for name in arm_names[1:]:
        deltas = [samples[name][r] - samples[baseline_name][r] for r in range(args.reps)]
        wins = sum(1 for d in deltas if d < 0)
        losses = sum(1 for d in deltas if d > 0)
        decided = wins + losses
        paired[name] = {"deltas_ms": deltas, "wins": wins, "losses": losses,
                        "ties": args.reps - decided,
                        "median_delta_ms": statistics.median(deltas),
                        "sign_test_p": sign_test(max(wins, losses), decided)}
        print(f"\npaired {name} vs {baseline_name} (per-repetition, "
              f"negative = {name} faster):")
        for index, delta in enumerate(deltas):
            print(f"  rep {index + 1}: {delta:+9.1f} ms  ({samples[name][index] / samples[baseline_name][index]:.3f}x)")
        verdict_paired = (f"{wins}W/{losses}L/{args.reps - decided}T, "
                          f"median {statistics.median(deltas):+.1f} ms, "
                          f"sign test p={paired[name]['sign_test_p']:.3f}")
        if args.reps < 5:
            verdict_paired += f" (at most p=0.25 is reachable at --reps {args.reps}; use 5 or more)"
        print(f"  {verdict_paired}")
    if paired:
        result["paired"] = paired

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
