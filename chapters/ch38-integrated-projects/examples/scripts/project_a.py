#!/usr/bin/env python3
"""Project A: generate P5 batch, run chapter 26 pipeline, verify delivered artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path

WIDTH, HEIGHT, FRAMES = 17, 13, 6


def generate_inputs(directory: Path, frames: int = FRAMES) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=False)
    paths: list[Path] = []
    for frame in range(frames):
        pixels = bytes((x * 13 + y * 17 + frame * 23) % 256
                       for y in range(HEIGHT) for x in range(WIDTH))
        path = directory / f"frame-{frame:03d}.pgm"
        path.write_bytes(f"P5\n{WIDTH} {HEIGHT}\n255\n".encode() + pixels)
        paths.append(path)
    return paths


def read_pgm(path: Path) -> tuple[int, int, bytes]:
    parts = path.read_bytes().split(b"\n", 3)
    if len(parts) != 4 or parts[0] != b"P5" or parts[2] != b"255":
        raise ValueError(f"unexpected PGM header: {path}")
    width, height = map(int, parts[1].split())
    if len(parts[3]) != width * height:
        raise ValueError(f"unexpected PGM payload length: {path}")
    return width, height, parts[3]


def read_f32(path: Path, count: int) -> tuple[float, ...]:
    payload = path.read_bytes()
    if len(payload) != 4 * count:
        raise ValueError(f"unexpected float tensor bytes: {path}")
    values = struct.unpack(f"<{count}f", payload)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"nonfinite float tensor: {path}")
    return values


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_outputs(inputs: Path, outputs: Path) -> tuple[dict[str, dict], float]:
    expected_names = {path.name for path in inputs.glob("*.pgm")}
    manifest_path = outputs / "manifest.tsv"
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != len(expected_names):
        raise ValueError("manifest row count differs from input count")
    manifest: dict[str, dict] = {}
    max_error = 0.0
    for index, row in enumerate(rows):
        if set(row) != {"index", "source", "width", "height", "preview", "tensor_f32_le"}:
            raise ValueError("manifest columns differ from contract")
        if int(row["index"]) != index:
            raise ValueError("manifest indices are not consecutive")
        name = row["source"]
        if name not in expected_names or name in manifest:
            raise ValueError("source missing or repeated in manifest")
        expected_width, expected_height = (WIDTH + 1) // 2, (HEIGHT + 1) // 2
        width, height = int(row["width"]), int(row["height"])
        if (width, height) != (expected_width, expected_height):
            raise ValueError("unexpected output shape")
        preview_path = outputs / row["preview"]
        tensor_path = outputs / row["tensor_f32_le"]
        pgm_width, pgm_height, preview = read_pgm(preview_path)
        if (pgm_width, pgm_height) != (width, height):
            raise ValueError("preview dimensions differ from manifest")
        tensor = read_f32(tensor_path, width * height)
        for byte, value in zip(preview, tensor):
            max_error = max(max_error, abs(value - (2.0 * byte / 255.0 - 1.0)))
        manifest[name] = {
            "width": width, "height": height,
            "preview": preview_path, "tensor": tensor_path,
            "preview_sha256": sha256(preview_path), "tensor_sha256": sha256(tensor_path),
        }
    if set(manifest) != expected_names or max_error > 1e-6:
        raise ValueError(f"artifact verification failed, max normalization error={max_error}")
    return manifest, max_error


def run_pipeline(binary: Path, outputs: Path, inputs: Path, log: Path,
                 env: dict[str, str] | None = None) -> None:
    completed = subprocess.run([str(binary), str(outputs), str(inputs)],
                               env=env, capture_output=True, text=True, check=False)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0 or "chapter 26 image pipeline: PASS" not in completed.stdout:
        raise RuntimeError(f"image pipeline failed with code {completed.returncode}; see {log}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binary = args.pipeline.resolve()
    if not binary.is_file():
        parser.error(f"pipeline executable missing: {binary}")
    root = args.output.resolve()
    if root.exists():
        parser.error(f"output path already exists; choose a fresh path: {root}")
    root.mkdir(parents=True)
    inputs, outputs = root / "input", root / "pipeline-output"
    generate_inputs(inputs)
    run_pipeline(binary, outputs, inputs, root / "pipeline.log")
    manifest, max_error = verify_outputs(inputs, outputs)
    report = {
        "project": "A", "status": "PASS", "frames": len(manifest),
        "input_shape": [HEIGHT, WIDTH], "output_shape": [(HEIGHT + 1) // 2, (WIDTH + 1) // 2],
        "max_tensor_vs_preview_error": max_error,
        "artifacts": {name: {key: value for key, value in entry.items()
                             if key.endswith("sha256")} for name, entry in manifest.items()},
        "pipeline_log": "pipeline.log",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"chapter 38 project A: PASS frames={len(manifest)} max_norm_error={max_error:.3g} output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
