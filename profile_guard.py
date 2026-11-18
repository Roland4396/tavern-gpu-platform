#!/usr/bin/env python3
"""Validate a workload contract without starting a GPU process."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def _run(argv: list[str]) -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def validate(stack_path: Path, workload: str) -> dict[str, Any]:
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    if int(stack.get("schema_version") or 0) != 1:
        raise RuntimeError("unsupported stack manifest schema")
    spec = (stack.get("workloads") or {}).get(workload)
    if not isinstance(spec, dict):
        raise RuntimeError(f"unknown workload: {workload}")
    platform_spec = dict(stack.get("platform") or {})
    if platform.machine() != "x86_64":
        raise RuntimeError(f"unsupported host architecture: {platform.machine()}")
    expected_gpus = int(spec.get("gpu_count") or 0)
    if expected_gpus < 0 or expected_gpus > int(platform_spec.get("gpu_count") or 0):
        raise RuntimeError(f"invalid GPU allocation for {workload}: {expected_gpus}")
    if not str(spec.get("launcher") or ""):
        raise RuntimeError(f"{workload} has no launcher contract")
    driver = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return {
        "status": "ok",
        "workload": workload,
        "kind": spec.get("kind"),
        "gpu_count": expected_gpus,
        "node": os.environ.get("GPU_RUNTIME_NODE", platform.node()),
        "driver_visible": bool(driver),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=Path, default=Path(__file__).with_name("stack.json"))
    parser.add_argument("workload")
    args = parser.parse_args(argv)
    print(json.dumps(validate(args.stack, args.workload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
