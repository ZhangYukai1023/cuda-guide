#!/usr/bin/env python3
"""Project D: split project A frames across two independent GPU processes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from project_a import FRAMES, generate_inputs, read_f32, read_pgm, run_pipeline, verify_outputs


def visible_physical_gpus() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    return sum(line.startswith("GPU ") for line in result.stdout.splitlines())


def compare_results(baseline: dict[str, dict], parallel: dict[str, dict]) -> float:
    if set(baseline) != set(parallel):
        raise ValueError("parallel output does not cover baseline input names")
    max_float_error = 0.0
    for name in baseline:
        a, b = baseline[name], parallel[name]
        aw, ah, ap = read_pgm(a["preview"])
        bw, bh, bp = read_pgm(b["preview"])
        if (aw, ah, ap) != (bw, bh, bp):
            raise ValueError(f"preview mismatch between one/two GPUs: {name}")
        x, y = read_f32(a["tensor"], aw * ah), read_f32(b["tensor"], aw * ah)
        max_float_error = max(max_float_error, *(abs(left - right) for left, right in zip(x, y)))
    if max_float_error > 1e-6:
        raise ValueError(f"one/two GPU tensor difference={max_float_error}")
    return max_float_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", default="0,1", help="two distinct physical GPU indices, e.g. 0,1")
    args = parser.parse_args()
    chosen = args.devices.split(",")
    if len(chosen) != 2 or not all(item.isdecimal() for item in chosen) or chosen[0] == chosen[1]:
        parser.error("--devices requires two distinct nonnegative indices")
    count = visible_physical_gpus()
    if count < 2 or any(int(item) >= count for item in chosen):
        print(f"SKIP project D: detected {count} physical GPU(s), requested {args.devices}")
        return 77
    binary = args.pipeline.resolve()
    if not binary.is_file():
        parser.error(f"pipeline executable missing: {binary}")
    root = args.output.resolve()
    if root.exists():
        parser.error(f"output path already exists; choose a fresh path: {root}")
    root.mkdir(parents=True)
    inputs = root / "input"
    paths = generate_inputs(inputs, FRAMES)
    baseline_out = root / "one-gpu-output"
    baseline_env = os.environ.copy()
    baseline_env["CUDA_VISIBLE_DEVICES"] = chosen[0]
    start = time.perf_counter()
    run_pipeline(binary, baseline_out, inputs, root / "one-gpu.log", baseline_env)
    baseline_seconds = time.perf_counter() - start
    baseline, _ = verify_outputs(inputs, baseline_out)
    partitions: list[tuple[Path, Path, Path, dict[str, str]]] = []
    for gpu_index, gpu in enumerate(chosen):
        part_input = root / f"gpu{gpu_index}-input"
        part_input.mkdir()
        for frame_index, path in enumerate(paths):
            if frame_index % 2 == gpu_index:
                shutil.copy2(path, part_input / path.name)
        part_output = root / f"gpu{gpu_index}-output"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        partitions.append((part_input, part_output, root / f"gpu{gpu_index}.log", env))
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_pipeline, binary, out, inp, log, env)
                   for inp, out, log, env in partitions]
        for future in futures:
            future.result()
    parallel_seconds = time.perf_counter() - start
    merged: dict[str, dict] = {}
    for inp, out, _, _ in partitions:
        subset, _ = verify_outputs(inp, out)
        for name, entry in subset.items():
            if name in merged:
                raise ValueError(f"duplicate source after merge: {name}")
            merged[name] = entry
    max_error = compare_results(baseline, merged)
    report = {
        "project": "D", "status": "PASS", "devices": chosen,
        "frames": FRAMES, "max_tensor_difference": max_error,
        "one_gpu_wall_seconds_single_run": baseline_seconds,
        "two_gpu_wall_seconds_single_run": parallel_seconds,
        "note": "Includes process startup, file IO and CPU reference; one run is not a stable speedup claim.",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"chapter 38 project D: PASS frames={FRAMES} max_difference={max_error:.3g} output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
