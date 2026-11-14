"""Probe a pinned VoxLock wrapper with an identity stub, never model weights.

uv run --with numpy==2.5.3 scripts/vad/probe_voxlock_bookkeeping.py --source <voxlock/core.py>
The expected source hash is checked before executing selected class/function
definitions. ONNX Runtime and model downloads are not required.
"""

import argparse
import ast
from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np


EXPECTED_SOURCE_SHA256 = "28049a0904482472a7892e1a357350d31418c3009fd2209f78007f73dc2afaca"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        parser.error("Source differs from reviewed VoxLock revision; re-audit it before changing the pin")
    tree = ast.parse(source.decode("utf-8"))
    names = {"Stream", "LiveStream", "loud_sample"}
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected} == names
    win2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(512) / 512)).astype(np.float32) ** 2
    scope = {"np": np, "deque": deque, "SR": 16000, "HOP": 128, "NFFT": 512, "PAD": 256, "WIN2": win2, "GAIN": -0.5}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(args.source), "exec"), scope)

    class IdentityEnroll:
        def run(self, *_):
            return np.zeros((1, 257, 1), np.float32), np.zeros((1, 512, 1), np.float32)

    class IdentityChunk:
        def run(self, _, inputs):
            wav = inputs["wav"][0]
            n = 1 + (len(wav) - 512) // 128
            frames = np.stack([wav[j * 128:j * 128 + 512] * win2 / -0.5 for j in range(n)])[None]
            return frames, inputs["h"], inputs["c"]

    class IdentityModel:
        enroll = IdentityEnroll()
        chunk = IdentityChunk()

        def stream(self, sample):
            return scope["Stream"](self, sample)

    model = IdentityModel()
    rows = []
    for n in [0, 1, 127, 128, 255, 256, 257, 511, 512, 513, 640, 1600, 48000, 48001]:
        x = np.sin(np.arange(n, dtype=np.float32) * 0.1) * 0.25
        stream = model.stream(np.zeros(48000, np.float32))
        pushed, flushed = stream.push(x), stream.flush()
        y = np.concatenate([pushed, flushed])
        row = {"input": n, "push_output": len(pushed), "flush_output": len(flushed), "total_output": len(y), "second_flush_output": len(stream.flush())}
        if len(y) >= n and n:
            row["trimmed_max_error"] = float(np.max(np.abs(y[:n] - x)))
        rows.append(row)
    live = scope["LiveStream"](model)
    live.need = 1  # force a switch to test bookkeeping, not enrollment quality
    live.energy.extend([-60.0] * 20)
    x = np.sin(np.arange(640, dtype=np.float32) * 0.1) * 0.25
    first = live.push(x)
    pending = live.skip
    tail = live.flush()
    forced = {"input": 640, "passthrough": len(first), "warm_samples_still_to_skip": pending, "flush_output": len(tail)}
    # These assertions characterize the reviewed reference, not desired product behavior.
    assert all(row["total_output"] == 0 for row in rows if 0 < row["input"] <= 256)
    assert next(row for row in rows if row["input"] == 640)["total_output"] == 1024
    assert forced == {"input": 640, "passthrough": 640, "warm_samples_still_to_skip": 384, "flush_output": 768}
    result = {"scope": "wrapper bookkeeping only; identity stub and forced switch; no neural inference", "numpy_version": np.__version__, "source_sha256": digest, "rows": rows, "forced_switch": forced}
    output = Path(__file__).resolve().parents[2] / "build" / "voxlock-bookkeeping-probe.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Observed {len(rows)} lengths plus forced live switch; reference edge cases reproduced. {output}")


if __name__ == "__main__":
    main()
