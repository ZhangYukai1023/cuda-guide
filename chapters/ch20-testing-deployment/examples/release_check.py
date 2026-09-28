#!/usr/bin/env python3
"""Build, test, and record a CUDA guide checkout without changing its toolchain."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(argv: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    record: dict[str, Any] = {"argv": argv, "timeout_seconds": timeout}
    try:
        result = subprocess.run(
            argv, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False
        )
        record.update(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        record.update(
            returncode=None,
            stdout=exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or "",
            stderr=exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or "",
            timed_out=True,
        )
    except OSError as exc:
        record.update(returncode=None, stdout="", stderr=str(exc), timed_out=False)
    return record


def cmake_settings(cache: Path) -> dict[str, str]:
    wanted = {"CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_COMPILER", "CMAKE_BUILD_TYPE"}
    found: dict[str, str] = {}
    if not cache.is_file():
        return found
    for line in cache.read_text(errors="replace").splitlines():
        if ":" not in line or "=" not in line:
            continue
        key, rest = line.split(":", 1)
        if key in wanted:
            found[key] = rest.split("=", 1)[1]
    return found


def parse_wall_medians(output: str) -> dict[str, float]:
    medians: dict[str, float] = {}
    for line in output.splitlines():
        match = re.search(r"\bn=(\d+)\b.*\bwall_ms=([0-9]+(?:\.[0-9]+)?)", line)
        if match:
            medians[match.group(1)] = float(match.group(2))
    return medians


def compare_baseline(current: dict[str, Any], baseline_path: Path, factor: float) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text())
    current_environment = current["environment"]
    old_environment = baseline.get("environment", {})
    same_gpu = current_environment.get("gpu", {}).get("stdout") == old_environment.get("gpu", {}).get("stdout")
    same_build = current_environment.get("cmake_settings") == old_environment.get("cmake_settings")
    if not (same_gpu and same_build):
        return {"status": "NOT_COMPARABLE", "reason": "GPU/driver or CMake settings differ"}
    old = baseline.get("performance", {}).get("wall_ms", {})
    new = current.get("performance", {}).get("wall_ms", {})
    rows: list[dict[str, Any]] = []
    for n in sorted(set(old) & set(new), key=int):
        old_ms = float(old[n])
        new_ms = float(new[n])
        if old_ms <= 0:
            continue
        rows.append({"n": int(n), "baseline_ms": old_ms, "current_ms": new_ms,
                     "ratio": new_ms / old_ms,
                     "needs_review": new_ms / old_ms > factor})
    if not rows:
        return {"status": "NOT_COMPARABLE", "reason": "no shared benchmark sizes"}
    return {"status": "REVIEW" if any(row["needs_review"] for row in rows) else "NO_FLAG",
            "threshold_factor": factor, "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, help="Built chapter 12 benchmark_vector binary")
    parser.add_argument("--sanitizer-binary", type=Path, help="Built safe-mode binary to check with memcheck")
    parser.add_argument("--baseline-report", type=Path, help="Prior report from same GPU and build settings")
    parser.add_argument("--regression-factor", type=float, default=1.15)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.regression_factor <= 1:
        parser.error("--regression-factor must be greater than 1")
    repo = args.repo.resolve()
    build = args.build_dir.resolve()
    if not repo.is_dir() or not build.is_dir():
        parser.error("--repo and --build-dir must already exist")
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo": str(repo), "build_dir": str(build),
        "environment": {"cmake_settings": cmake_settings(build / "CMakeCache.txt")},
        "steps": {}, "performance": {"wall_ms": {}},
    }
    env = report["environment"]
    env["git_commit"] = run_command(["git", "rev-parse", "HEAD"], repo, 10)
    env["git_status"] = run_command(["git", "status", "--short"], repo, 10)
    env["gpu"] = run_command(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], repo, 20
    )
    nvcc = env["cmake_settings"].get("CMAKE_CUDA_COMPILER", "nvcc")
    env["nvcc"] = run_command([nvcc, "--version"], repo, 20)

    steps = report["steps"]
    steps["build"] = run_command(["cmake", "--build", str(build), "-j2"], repo, 600)
    build_ok = steps["build"]["returncode"] == 0
    if build_ok:
        steps["ctest"] = run_command(
            ["ctest", "--test-dir", str(build), "--output-on-failure"], repo, 900
        )
        if args.benchmark:
            steps["benchmark"] = run_command([str(args.benchmark.resolve())], repo, 120)
            if steps["benchmark"]["returncode"] == 0:
                report["performance"]["wall_ms"] = parse_wall_medians(steps["benchmark"]["stdout"])
        if args.sanitizer_binary:
            sanitizer = shutil.which("compute-sanitizer")
            if sanitizer:
                steps["memcheck"] = run_command(
                    [sanitizer, "--tool", "memcheck", "--error-exitcode", "86",
                     str(args.sanitizer_binary.resolve()), "safe"], repo, 180
                )
            else:
                steps["memcheck"] = {"returncode": None, "stderr": "compute-sanitizer unavailable"}
    if args.baseline_report and report["performance"]["wall_ms"]:
        try:
            report["baseline_comparison"] = compare_baseline(
                report, args.baseline_report.resolve(), args.regression_factor
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["baseline_comparison"] = {"status": "ERROR", "reason": str(exc)}

    required = [env["gpu"], env["nvcc"], steps.get("build"), steps.get("ctest")]
    if args.benchmark:
        required.append(steps.get("benchmark"))
    if args.sanitizer_binary:
        required.append(steps.get("memcheck"))
    report["ok"] = all(item is not None and item.get("returncode") == 0 for item in required)
    if steps.get("ctest") and "No tests were found" in (
        steps["ctest"].get("stdout", "") + steps["ctest"].get("stderr", "")
    ):
        report["ok"] = False
    if args.benchmark and not report["performance"]["wall_ms"]:
        report["ok"] = False
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"report={out} status={'PASS' if report['ok'] else 'FAIL'}")
    if "baseline_comparison" in report:
        print(f"performance_comparison={report['baseline_comparison']['status']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
