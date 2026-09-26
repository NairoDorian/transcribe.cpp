#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# ///
"""Find (and optionally fix) text I/O without an explicit encoding.

On Windows, Python's default text encoding is the ANSI code page (cp1252 on
Western systems), not UTF-8. Every `open(path)`, `Path.read_text()` or
`subprocess.run(..., text=True)` without `encoding=` therefore decodes UTF-8
manifests, transcripts and CLI output as cp1252: "é" becomes "Ã©", "ü" becomes
"Ã¼", Cyrillic/CJK turn into garbage, and some characters raise
UnicodeDecodeError. WER references and hypotheses then get compared mangled.

The rule for this repo: every text-mode file open and every text-mode
subprocess states its encoding. Files are UTF-8; subprocess output is decoded
as UTF-8 with errors="replace" (transcribe.cpp tools print UTF-8).

    uv run scripts/ci/check_text_encoding.py            # report, exit 1 if any
    uv run scripts/ci/check_text_encoding.py --fix      # rewrite in place
    uv run scripts/ci/check_text_encoding.py PATH ...   # limit to files/dirs

Detection is AST-based (multi-line calls are handled). Binary modes ("rb",
"wb", ...) and non-text openers (wave, tarfile, gzip, zipfile, PIL, soundfile,
os.open, urlopen, ...) are skipped.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = [REPO / "scripts", REPO / "bindings" / "python" / "transcribe_cpp"]
SKIP_PARTS = {".venv", "site-packages", "node_modules", "__pycache__", "build"}
NON_TEXT_RECEIVERS = {"wave", "tarfile", "gzip", "bz2", "lzma", "zipfile", "Image", "sf", "soundfile",
                      "os", "io", "tempfile", "shelve", "dbm", "webbrowser", "urllib", "request", "h5py",
                      "np", "numpy", "torch", "safetensors", "gguf", "zf", "tf", "tar", "zip", "z"}
SUBPROCESS_FUNCS = {"run", "Popen", "check_output", "check_call", "call"}


def _mode_of(call: ast.Call, pos_index: int) -> str | None:
    for kw in call.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    if len(call.args) > pos_index and isinstance(call.args[pos_index], ast.Constant) \
            and isinstance(call.args[pos_index].value, str):
        return call.args[pos_index].value
    if len(call.args) > pos_index or any(kw.arg == "mode" for kw in call.keywords):
        return "?"  # dynamic mode: cannot tell, leave alone
    return "r"


def _has_kw(call: ast.Call, *names: str) -> bool:
    return any(kw.arg in names for kw in call.keywords) or any(
        isinstance(kw.value, ast.Dict) and kw.arg is None for kw in call.keywords)


def _receiver_name(func: ast.Attribute) -> str | None:
    v = func.value
    while isinstance(v, ast.Attribute):
        v = v.value
    return v.id if isinstance(v, ast.Name) else None


def findings(tree: ast.AST):
    """Yield (call, insertion) for every call that needs an encoding."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        # builtin open(file, mode="r", ...)
        if isinstance(f, ast.Name) and f.id == "open":
            m = _mode_of(node, 1)
            if m not in ("?",) and "b" not in m and not _has_kw(node, "encoding"):
                yield node, 'encoding="utf-8"'
        elif isinstance(f, ast.Attribute):
            recv = _receiver_name(f)
            if f.attr in ("read_text", "write_text"):
                # Path.read_text(encoding=None) / write_text(data, encoding=None)
                limit = 0 if f.attr == "read_text" else 1
                if len(node.args) <= limit and not _has_kw(node, "encoding"):
                    yield node, 'encoding="utf-8"'
            elif f.attr == "open" and recv not in NON_TEXT_RECEIVERS:
                # Path.open(mode="r", ...) — first positional is the mode
                m = _mode_of(node, 0)
                if m not in ("?",) and "b" not in m and not _has_kw(node, "encoding"):
                    yield node, 'encoding="utf-8"'
            elif f.attr in SUBPROCESS_FUNCS and recv == "subprocess":
                text = any(kw.arg in ("text", "universal_newlines") and isinstance(kw.value, ast.Constant)
                           and kw.value.value is True for kw in node.keywords)
                if text and not _has_kw(node, "encoding"):
                    extra = 'encoding="utf-8"' if _has_kw(node, "errors") else 'encoding="utf-8", errors="replace"'
                    yield node, extra


def _insert(src_lines: list[str], call: ast.Call, text: str) -> None:
    """Insert `text` as the last argument of `call` (edits src_lines in place)."""
    ln, col = call.end_lineno - 1, call.end_col_offset - 1  # position of the closing ')'
    line = src_lines[ln]
    assert line[col] == ")", (line, col)
    # previous significant char before ')'
    j, k = ln, col - 1
    while True:
        while k >= 0 and src_lines[j][k] in " \t\r\n":
            k -= 1
        if k >= 0:
            break
        j -= 1
        k = len(src_lines[j]) - 1
    prev = src_lines[j][k]
    sep = "" if prev in ",(" else ", "
    if prev == ",":
        sep = " "
    src_lines[ln] = line[:col] + sep + text + line[col:]


def process(path: Path, fix: bool) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    found = sorted(findings(tree), key=lambda t: (t[0].end_lineno, t[0].end_col_offset), reverse=True)
    if not found:
        return []
    report = [(c.lineno, ast.get_source_segment(src, c).splitlines()[0][:100]) for c, _ in found]
    if fix:
        lines = src.split("\n")
        for call, text in found:  # bottom-up so earlier offsets stay valid
            _insert(lines, call, text)
        new = "\n".join(lines)
        compile(new, str(path), "exec")  # never write a file that no longer compiles
        path.write_text(new, encoding="utf-8", newline="")
    return sorted(report)


def main(argv: list[str]) -> int:
    fix = "--fix" in argv
    targets = [Path(a) for a in argv if not a.startswith("--")] or DEFAULT_ROOTS
    files: list[Path] = []
    for t in targets:
        if t.is_file():
            files.append(t)
        elif t.is_dir():
            files += [p for p in t.rglob("*.py") if not (set(p.parts) & SKIP_PARTS)]
    total = 0
    for p in sorted(files):
        rep = process(p, fix)
        for ln, seg in rep:
            print(f"{p.relative_to(REPO) if p.is_relative_to(REPO) else p}:{ln}: {seg}")
        total += len(rep)
    verb = "fixed" if fix else "found"
    print(f"\n{verb} {total} text I/O call(s) without an explicit encoding in {len(files)} file(s)")
    return 0 if (fix or total == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
