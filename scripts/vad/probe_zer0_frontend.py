"""Probe the paired app's framing and dry-path signal delay; not a speed benchmark.

Run with uv run. App sources are copied to ignored build/ so a probe does not
edit or compile the complete app. No model downloads or microphone access.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zer0-root", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, help="Optional Cargo artifact cache")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = repo / "build" / "zer0-frontend-probe"
    sources = args.zer0_root.resolve() / "src-tauri" / "src" / "audio_toolkit" / "audio"
    for name in ("resampler.rs", "denoise.rs"):
        if not (sources / name).is_file():
            parser.error(f"Missing paired app source: {sources / name}")
    marker = output / ".zer0-frontend-probe"
    if output.exists() and any(output.iterdir()) and not marker.exists():
        parser.error(f"Refusing to overwrite an unrelated directory: {output}")
    (output / "src").mkdir(parents=True, exist_ok=True)
    marker.touch()
    hashes = {}
    for name in ("resampler.rs", "denoise.rs"):
        data = (sources / name).read_bytes()
        (output / "src" / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    shutil.copyfile(Path(__file__).with_suffix(".rs"), output / "src" / "main.rs")
    (output / "Cargo.toml").write_text('''[package]
name = "zer0-frontend-probe"
version = "0.0.0"
edition = "2021"
publish = false

[workspace]

[dependencies]
rubato = "=5.0.1"
nnnoiseless = { version = "=0.5.2", default-features = false }
log = "0.4"
''', encoding="utf-8")
    command = ["cargo", "run", "--offline", "--release", "--manifest-path", str(output / "Cargo.toml")]
    if args.target_dir:
        command.extend(["--target-dir", str(args.target_dir.resolve())])
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    (output / "results.csv").write_text(result.stdout, encoding="utf-8")
    (output / "build.log").write_text(result.stderr, encoding="utf-8")
    manifest = {
        "kind": "synthetic framing and impulse alignment; not compute latency or WER",
        "host": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "source_root": str(sources),
        "source_sha256": hashes,
        "cargo_lock_sha256": hashlib.sha256((output / "Cargo.lock").read_bytes()).hexdigest(),
        "input": "2 seconds, 0.5 impulse at 100 ms, synthetic 10 ms deliveries",
        "denoise": "strength=0, probability gate=0; network still runs",
    }
    (output / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(result.stdout, end="")
    print(f"Artifacts: {output}")


if __name__ == "__main__":
    main()
