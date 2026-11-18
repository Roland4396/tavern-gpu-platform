#!/usr/bin/env python3
"""Prepare an incremental GPU platform release from an active baseline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def replace_component(
    source: Path,
    output: Path,
    release_id: str,
    component: str,
    store_path: Path,
) -> dict[str, Any]:
    value = json.loads(source.read_text(encoding="utf-8"))
    components = dict(value.get("components") or {})
    if component not in components:
        raise KeyError(f"unknown component: {component}")
    if store_path.parent != Path("/nix/store") or not store_path.is_dir():
        raise ValueError(f"invalid Nix store path: {store_path}")
    updated = dict(components[component])
    updated["store_path"] = str(store_path)
    updated["cache_published"] = False
    components[component] = updated
    value.update(
        {
            "release_id": release_id,
            "created_at": int(time.time()),
            "components": components,
        }
    )
    atomic_json(output, value)
    return value


def release_profiles(
    active: Path,
    output: Path,
    release_id: str,
    replacements: dict[str, Path],
) -> dict[str, Any]:
    value = json.loads(active.read_text(encoding="utf-8"))
    profiles = dict(value.get("profiles") or {})
    if not profiles:
        raise ValueError("active profile map is empty")
    for name, store_path in replacements.items():
        if name not in profiles:
            raise KeyError(f"unknown profile: {name}")
        if store_path.parent != Path("/nix/store") or not store_path.is_dir():
            raise ValueError(f"invalid Nix store path: {store_path}")
        profiles[name] = str(store_path)
    result = {
        "gpu_stack_release_id": release_id,
        "gpu_stack_profiles": profiles,
    }
    atomic_json(output, result)
    return result


def parse_replacements(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid profile replacement: {value}")
        result[name] = Path(raw_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    component_parser = subparsers.add_parser("component")
    component_parser.add_argument("--source", type=Path, required=True)
    component_parser.add_argument("--output", type=Path, required=True)
    component_parser.add_argument("--release-id", required=True)
    component_parser.add_argument("--component", required=True)
    component_parser.add_argument("--store-path", type=Path, required=True)

    profile_parser = subparsers.add_parser("profiles")
    profile_parser.add_argument("--active", type=Path, required=True)
    profile_parser.add_argument("--output", type=Path, required=True)
    profile_parser.add_argument("--release-id", required=True)
    profile_parser.add_argument("--profile", action="append", default=[])

    args = parser.parse_args(argv)
    if args.command == "component":
        result = replace_component(
            args.source,
            args.output,
            args.release_id,
            args.component,
            args.store_path,
        )
    else:
        result = release_profiles(
            args.active,
            args.output,
            args.release_id,
            parse_replacements(args.profile),
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
