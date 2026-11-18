#!/usr/bin/env python3
"""Atomically activate or roll back a set of Nix workload profiles."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any


def _validate_name(name: str) -> None:
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name):
        raise ValueError(f"invalid profile name: {name!r}")


def _validate_store_path(path: Path) -> Path:
    raw = path.absolute()
    if raw.parent != Path("/nix/store"):
        raise ValueError(f"profile target is not a direct Nix store path: {path}")
    if not raw.is_dir():
        raise FileNotFoundError(raw)
    return raw


def _atomic_link(link: Path, target: Path) -> None:
    temporary = link.parent / f".{link.name}.{uuid.uuid4().hex}"
    temporary.symlink_to(target)
    os.replace(temporary, link)


def _write_state(root: Path, profiles: dict[str, Path]) -> None:
    state = {
        "schema_version": 1,
        "profiles": {name: str(path) for name, path in sorted(profiles.items())},
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=".active.", suffix=".json", dir=root)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o600)
        os.replace(temporary, root / "active.json")
    finally:
        temporary.unlink(missing_ok=True)


def activate(root: Path, requested: dict[str, Path]) -> dict[str, str]:
    targets: dict[str, Path] = {}
    for name, path in requested.items():
        _validate_name(name)
        targets[name] = _validate_store_path(path)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    old_current: dict[str, Path | None] = {}
    old_previous: dict[str, Path | None] = {}
    changed: list[str] = []
    try:
        for name, target in sorted(targets.items()):
            directory = root / name
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            current = directory / "current"
            previous = directory / "previous"
            existing = current.resolve(strict=True) if current.is_symlink() else None
            if current.exists() and not current.is_symlink():
                raise RuntimeError(f"refusing to replace non-symlink profile: {current}")
            prior = previous.resolve(strict=True) if previous.is_symlink() else None
            if previous.exists() and not previous.is_symlink():
                raise RuntimeError(f"refusing to replace non-symlink profile: {previous}")
            old_current[name] = existing
            old_previous[name] = prior
            if existing == target:
                continue
            changed.append(name)
            if existing is not None:
                _atomic_link(previous, existing)
            _atomic_link(current, target)
        active = {
            directory.name: (directory / "current").resolve(strict=True)
            for directory in root.iterdir()
            if directory.is_dir() and (directory / "current").is_symlink()
        }
        _write_state(root, active)
    except BaseException:
        for name in reversed(changed):
            current = root / name / "current"
            previous = root / name / "previous"
            for link, target in (
                (current, old_current[name]),
                (previous, old_previous[name]),
            ):
                if target is None:
                    link.unlink(missing_ok=True)
                else:
                    _atomic_link(link, target)
        raise
    return {name: str(path) for name, path in sorted(active.items())}


def rollback(root: Path, names: list[str]) -> dict[str, str]:
    requested: dict[str, Path] = {}
    for name in names:
        _validate_name(name)
        previous = root / name / "previous"
        if not previous.is_symlink():
            raise RuntimeError(f"profile has no previous generation: {name}")
        requested[name] = previous.resolve(strict=True)
    return activate(root, requested)


def _load_profiles(path: Path) -> dict[str, Path]:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("profiles"), dict):
        raw = raw["profiles"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("profiles JSON must be a non-empty object")
    return {str(name): Path(str(target)) for name, target in raw.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/home/worker/.local/profiles/gpu-runtime"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    activate_parser = subparsers.add_parser("activate")
    activate_parser.add_argument("--profiles-json", type=Path, required=True)
    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("profiles", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "activate":
        result = activate(args.root, _load_profiles(args.profiles_json))
    else:
        result = rollback(args.root, args.profiles)
    print(json.dumps({"status": "ok", "profiles": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
