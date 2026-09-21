"""Render a benchmark summary.json as Markdown and CSV.

Usage:
    uv run --no-project scripts/bench/report.py --summary <dir>/summary.json

The suite writes JSON only; this turns the same data into the two forms a
reviewer actually reads (a table to paste into a discussion, a CSV to diff
between runs). It reads nothing but the summary, so it works unchanged on the
fork's and the upstream tree's output, which is what makes the two comparable.

Per-stage columns come from the streaming chunk instrumentation. They are
blank for batch cases, which have no per-chunk pipeline to break down, and for
app summaries whose staged library lacks the instrumentation.

The app runner (Handy_benchmarks/scripts/bench-stt.ts) deliberately emits the
same field names — `warm_mean_timings`, `stage_metrics`, and a `bound_backend`
alias for `backend` — so one reporter renders the native and app summaries
alike instead of two drifting implementations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Stage order mirrors emit_streaming_chunk's own report so a reader can map a
# column straight onto the native line.
STAGES = ("total", "graph_build", "sched_alloc", "graph_compute", "readback", "cache_rot", "decoder", "other")
TIMING_KEYS = ("mel_ms", "encode_ms", "decode_ms")


def _fmt(value, digits=1):
    return "" if value is None else f"{value:.{digits}f}"


def rows_from(summary):
    for case in summary.get("cases", []):
        stages = (case.get("stage_metrics") or {}).get("stages_ms") or {}
        # suite.py stores the warm timing means under "timings"; accept the
        # pipeline.py field name too so either file can be rendered directly.
        timings = case.get("timings") or case.get("warm_mean_timings") or {}
        row = {"case": case.get("case", ""),
               # App summaries call this bound_backend; accept either name.
               "backend": case.get("backend") or case.get("bound_backend", ""),
               "model_bytes": case.get("model_bytes", ""),
               "threads": case.get("threads", ""),
               "stream_chunk_ms": case.get("stream_chunk_ms", ""),
               "warm_mean_ms": case.get("warm_mean_ms"),
               "warm_mean_rtf": case.get("warm_mean_rtf"),
               "ratio_to_baseline": case.get("ratio_to_baseline"),
               "chunks": (case.get("stage_metrics") or {}).get("chunks", "")}
        for key in TIMING_KEYS:
            row[key] = timings.get(key)
        for stage in STAGES:
            stats = stages.get(stage) or {}
            row[f"chunk_{stage}_mean_ms"] = stats.get("mean")
            row[f"chunk_{stage}_p95_ms"] = stats.get("p95")
        row["text"] = case.get("text", "")
        yield row


# Ratios and RTFs need more resolution than the millisecond columns: at one
# decimal a 4% regression and a 14% one both render as "+0.1".
CSV_DIGITS = {"warm_mean_rtf": 3, "ratio_to_baseline": 3}


def write_csv(path, rows):
    fieldnames = (["case", "backend", "model_bytes", "threads", "stream_chunk_ms",
                   "warm_mean_ms", "warm_mean_rtf", "ratio_to_baseline", "chunks"]
                  + list(TIMING_KEYS)
                  + [f"chunk_{stage}_{stat}_ms" for stage in STAGES for stat in ("mean", "p95")]
                  + ["text"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (_fmt(row[key], CSV_DIGITS.get(key, 1)) if isinstance(row[key], float)
                                   else row[key])
                             for key in fieldnames})


def write_markdown(path, summary, rows):
    lines = ["# Benchmark report", ""]
    lines.append(f"- Machine: `{summary.get('machine', 'unknown')}`")
    lines.append(f"- Policy: {summary.get('policy', '')}")
    if summary.get("library"):
        lines.append(f"- Library: `{summary['library']}`")
    if summary.get("baseline"):
        lines.append(f"- Baseline: `{summary['baseline']}` (budget {summary.get('max_regression', '')})")
    lines += ["", "Times are milliseconds; lower is better. Warm mean is runs 2-3 with run 1",
              "discarded as warm-up. `mel`/`encode`/`decode` are the native result-stage timers;",
              "the `chunk_*` columns are per-chunk pipeline stages averaged over the same runs.", ""]

    header = ["case", "backend", "warm", "rtf", "vs base", "mel", "encode", "decode", "chunks",
              "chunk total", "graph_build", "sched_alloc", "graph_compute", "readback", "cache_rot", "decoder", "other"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in rows:
        lines.append("| " + " | ".join([
            row["case"], str(row["backend"]), _fmt(row["warm_mean_ms"]), _fmt(row["warm_mean_rtf"], 3),
            _fmt(row["ratio_to_baseline"], 3),
            _fmt(row["mel_ms"]), _fmt(row["encode_ms"]), _fmt(row["decode_ms"]), str(row["chunks"]),
            _fmt(row["chunk_total_mean_ms"]), _fmt(row["chunk_graph_build_mean_ms"]),
            _fmt(row["chunk_sched_alloc_mean_ms"]), _fmt(row["chunk_graph_compute_mean_ms"]),
            _fmt(row["chunk_readback_mean_ms"]), _fmt(row["chunk_cache_rot_mean_ms"]),
            _fmt(row["chunk_decoder_mean_ms"]), _fmt(row["chunk_other_mean_ms"])]) + " |")

    missing = summary.get("missing") or []
    if missing:
        lines += ["", "## Missing models", ""] + [f"- `{name}` (not installed; never downloaded)" for name in missing]
    failures = summary.get("failures") or []
    lines += ["", "## Failures", ""]
    lines += [f"- `{f.get('case', '?')}`: {f.get('error') or f.get('detail') or f.get('kind') or 'failure'}"
              for f in failures] if failures else ["", "None."]
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--md", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = list(rows_from(summary))
    md_path = args.md or args.summary.with_name("report.md")
    csv_path = args.csv or args.summary.with_name("report.csv")
    # Called programmatically by suite.py as well as by hand, so do not require
    # the caller to have created the destination directory first.
    for path in (md_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(md_path, summary, rows)
    write_csv(csv_path, rows)
    print(f"wrote {md_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
