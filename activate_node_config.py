#!/usr/bin/env python3
"""Attach the active platform profile metadata to node.json atomically."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def update_config(config: dict[str, Any], stack_path: Path, profile_root: Path) -> dict[str, Any]:
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    workloads = dict(stack.get("workloads") or {})
    configured = dict(config.get("workloads") or {})
    missing = sorted(set(workloads) - set(configured))
    if missing:
        raise RuntimeError(f"node config is missing stack workloads: {', '.join(missing)}")
    result = dict(config)
    result["stack_manifest"] = str(stack_path)
    updated_workloads = dict(configured)
    for name, workload_contract in workloads.items():
        spec = dict(updated_workloads[name])
        if "gpu_count" in workload_contract:
            spec["gpu_count"] = int(workload_contract["gpu_count"])
        profile_path = profile_root / name / "current"
        profile_command = str(workload_contract.get("profile_command") or "")
        if profile_command:
            executable = profile_path / "bin" / profile_command
            if not executable.is_file():
                if bool(workload_contract.get("profile_optional")) and spec.get("command"):
                    spec.pop("profile_path", None)
                else:
                    raise RuntimeError(f"profile executable is missing: {executable}")
            else:
                spec["profile_path"] = str(profile_path)
                old_command = [str(item) for item in spec.get("command") or []]
                if name == "graph":
                    if old_command and old_command[0] == str(executable):
                        preserved_args = old_command[1:]
                    else:
                        preserved_args = old_command[2:] if len(old_command) > 2 else []
                    preserved_args = [item for item in preserved_args if item != "--cpu"]
                    if int(workload_contract.get("gpu_count") or 0) == 0:
                        preserved_args.insert(0, "--cpu")
                else:
                    preserved_args = []
                spec["command"] = [str(executable), *preserved_args]
        else:
            spec["profile_path"] = str(profile_path)
        updated_workloads[name] = spec
    result["workloads"] = updated_workloads
    return result


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("/home/worker/.config/gpu-runtime/node.json"))
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("/home/worker/.local/profiles/gpu-runtime"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = json.loads(args.config.read_text(encoding="utf-8"))
    updated = update_config(current, args.stack, args.profile_root)
    if not args.check:
        atomic_write(args.config, updated)
    print(json.dumps({"status": "ok", "changed": updated != current, "config": str(args.config)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
