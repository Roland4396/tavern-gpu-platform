#!/usr/bin/env python3
"""Snapshot the canonical GPU runtime components into signed Nix store paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


HOME = Path("/home/worker")
DEFAULT_STAGE = HOME / "tmp/gpu-platform-components"
DEFAULT_OUTPUT = HOME / ".local/state/gpu-runtime/releases/components.json"
DEFAULT_CATALOG = HOME / ".local/state/gpu-runtime/releases/inventory-full.json"
TEXT_SUFFIXES = {"", ".json", ".pth", ".py", ".cfg", ".ini", ".sh", ".txt"}


def run(argv: list[str], *, timeout: int = 7200) -> str:
    return subprocess.run(
        argv,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        timeout=timeout,
    ).stdout.strip()


def remove_caches(root: Path) -> None:
    for directory in list(root.rglob("__pycache__")):
        if directory.is_dir():
            shutil.rmtree(directory)
    for pattern in ("*.pyc", "*.pyo"):
        for path in root.rglob(pattern):
            path.unlink(missing_ok=True)


def hardlink_tree(source: Path, target: Path) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    run(["cp", "-al", str(source), str(target)])
    remove_caches(target)


def rsync_tree(source: Path, target: Path, excludes: tuple[str, ...]) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    command = ["rsync", "-a", "--delete"]
    for pattern in excludes:
        command.extend(["--exclude", pattern])
    command.extend([f"{source}/", f"{target}/"])
    run(command)
    remove_caches(target)


def replace_text(root: Path, old: str, new: str) -> int:
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = text.replace(old, new)
        if updated == text:
            continue
        temporary = path.with_name(path.name + ".gpu-platform.tmp")
        temporary.write_text(updated, encoding="utf-8")
        temporary.chmod(path.stat().st_mode & 0o7777)
        os.replace(temporary, path)
        changed += 1
    return changed


def component_name(asset: dict[str, Any]) -> str:
    owned_id = str(asset.get("id") or "")
    if owned_id.startswith("owned:"):
        stable = re.sub(r"[^a-zA-Z0-9_-]+", "-", owned_id.removeprefix("owned:")).strip("-").lower()
        if stable:
            return stable
    kind = re.sub(r"[^a-z0-9]+", "-", str(asset.get("kind") or "asset").lower()).strip("-")
    path = str(asset.get("path") or asset.get("id") or "asset")
    basename = re.sub(r"[^a-zA-Z0-9]+", "-", Path(path).name).strip("-").lower() or "root"
    digest = hashlib.sha256(f"{kind}\0{path}".encode()).hexdigest()[:12]
    return f"{kind}-{basename[:36]}-{digest}"


def merge_system_site_packages(environment: Path) -> int:
    pyvenv = environment / "pyvenv.cfg"
    python = environment / "bin/python"
    if not pyvenv.is_file() or not python.exists():
        return 0
    value = pyvenv.read_text(encoding="utf-8")
    if "include-system-site-packages = true" not in value.lower():
        return 0
    output = run(
        [
            str(python),
            "-c",
            "import json,site; print(json.dumps(site.getsitepackages()))",
        ],
        timeout=120,
    )
    destinations = sorted(environment.glob("lib/python*/site-packages"))
    if not destinations:
        raise RuntimeError(f"cannot locate site-packages in {environment}")
    destination = destinations[0]
    merged = 0
    for raw in json.loads(output):
        source = Path(str(raw))
        try:
            source.relative_to(environment)
            continue
        except ValueError:
            pass
        if source.is_dir():
            run(["rsync", "-a", "--ignore-existing", f"{source}/", f"{destination}/"])
            merged += 1
    updated = re.sub(
        r"(?im)^include-system-site-packages\s*=\s*true\s*$",
        "include-system-site-packages = false",
        value,
    )
    temporary = pyvenv.with_suffix(".cfg.tmp")
    temporary.write_text(updated, encoding="utf-8")
    os.replace(temporary, pyvenv)
    return merged


class ComponentBuilder:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stage = args.stage
        self.catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
        self.components: dict[str, dict[str, Any]] = {}
        if args.output.is_file():
            try:
                existing = json.loads(args.output.read_text(encoding="utf-8"))
                if existing.get("release_id") == args.release_id:
                    self.components = dict(existing.get("components") or {})
            except (OSError, ValueError, TypeError, AttributeError):
                pass

    def existing(self, name: str) -> Path | None:
        raw = str((self.components.get(name) or {}).get("store_path") or "")
        path = Path(raw) if raw else None
        return path if path is not None and path.is_dir() else None

    def ensure_published(self, name: str, store_path: Path) -> None:
        # Legacy manifests predate this field and were written only after a
        # successful copy, so a missing value is already published.
        if (self.components.get(name) or {}).get("cache_published", True):
            return
        run(
            [
                self.args.nix,
                "store",
                "sign",
                "--key-file",
                str(self.args.signing_key),
                "--recursive",
                str(store_path),
            ]
        )
        run([self.args.nix, "copy", "--to", self.args.cache_uri, str(store_path)])
        self.components[name]["cache_published"] = True
        self.write_manifest()

    def add(self, name: str, path: Path, metadata: dict[str, Any] | None = None) -> Path:
        store_path = Path(run([self.args.nix, "store", "add", "--name", f"gpu-{name}", str(path)]))
        run(
            [
                self.args.nix,
                "store",
                "sign",
                "--key-file",
                str(self.args.signing_key),
                "--recursive",
                str(store_path),
            ]
        )
        info = json.loads(
            run([self.args.nix, "path-info", "--json", "--json-format", "1", str(store_path)])
        )
        self.components[name] = {
            "store_path": str(store_path),
            "path_info": info.get(str(store_path), info),
            "cache_published": False,
            **(metadata or {}),
        }
        self.write_manifest()
        self.ensure_published(name, store_path)
        return store_path

    def add_metadata(self, name: str, metadata: dict[str, Any]) -> None:
        self.components[name] = metadata
        self.write_manifest()

    def annotate(self, name: str, metadata: dict[str, Any]) -> None:
        current = dict(self.components.get(name) or {})
        current.update(metadata)
        self.components[name] = current
        self.write_manifest()

    def write_manifest(self) -> None:
        value = {
            "schema_version": 1,
            "release_id": self.args.release_id,
            "created_at": int(time.time()),
            "builder": os.uname().nodename,
            "catalog_path": str(self.args.catalog),
            "catalog_sha256": hashlib.sha256(self.args.catalog.read_bytes()).hexdigest(),
            "components": self.components,
        }
        self.args.output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=".components.", suffix=".json", dir=self.args.output.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(value, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self.args.output)
        finally:
            temporary.unlink(missing_ok=True)

    def build(self) -> None:
        if self.stage.exists():
            shutil.rmtree(self.stage)
        self.stage.mkdir(mode=0o700, parents=True)
        try:
            assets = list(self.catalog.get("runtime_assets") or [])
            if not assets:
                raise RuntimeError("catalog contains no runtime_assets")

            source_replacements: dict[str, str] = {}
            for asset in assets:
                if asset.get("kind") not in {"source-tree", "runtime-tree"}:
                    continue
                name = component_name(asset)
                existing = self.existing(name)
                if existing is not None:
                    self.ensure_published(name, existing)
                    self.annotate(name, {"asset_id": asset["id"], "asset_kind": asset["kind"], "asset_path": str(asset["path"])})
                    source_replacements[str(asset["path"])] = str(existing)
                    continue
                source = Path(str(asset["path"]))
                if not source.is_dir():
                    raise FileNotFoundError(source)
                target = self.stage / name
                rsync_tree(
                    source,
                    target,
                    (".git", ".venv", "models", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache", "outputs", "logs", "node_modules"),
                )
                metadata: dict[str, Any] = {"asset_id": asset["id"], "asset_kind": asset["kind"], "asset_path": str(source)}
                if (source / "models").is_dir():
                    (target / "models").symlink_to(source / "models")
                    metadata["external_models"] = str(source / "models")
                stored = self.add(name, target, metadata)
                source_replacements[str(source)] = str(stored)

            for asset in assets:
                if asset.get("kind") != "python-environment":
                    continue
                name = component_name(asset)
                existing = self.existing(name)
                if existing is not None:
                    self.ensure_published(name, existing)
                    self.annotate(name, {"asset_id": asset["id"], "asset_kind": asset["kind"], "asset_path": str(asset["path"]), "pip_freeze_sha256": asset.get("pip_freeze_sha256")})
                    continue
                source = Path(str(asset["path"]))
                if not source.is_dir():
                    raise FileNotFoundError(source)
                target = self.stage / name
                hardlink_tree(source, target)
                merged = merge_system_site_packages(target)
                replacements = sum(replace_text(target, old, new) for old, new in source_replacements.items())
                self.add(
                    name,
                    target,
                    {
                        "asset_id": asset["id"],
                        "asset_kind": asset["kind"],
                        "asset_path": str(source),
                        "pip_freeze_sha256": asset.get("pip_freeze_sha256"),
                        "source_path_replacements": replacements,
                        "system_site_package_roots_merged": merged,
                    },
                )

            for asset in assets:
                if asset.get("kind") not in {"runtime-config", "model-asset"}:
                    continue
                name = component_name(asset)
                if name in self.components:
                    continue
                # Model files stay on OSSFS and configuration can contain
                # secrets.  Both are governed by hashes in the release
                # catalog, never duplicated into the world-readable Nix store.
                self.add_metadata(
                    name,
                    {
                        "asset_id": asset["id"],
                        "asset_kind": asset["kind"],
                        "asset_path": str(asset["path"]),
                        "mode": "external" if asset["kind"] == "model-asset" else "metadata-only",
                        "sha256": asset.get("sha256"),
                        "metadata_sha256": asset.get("metadata_sha256"),
                        "size": asset.get("size"),
                    },
                )
        finally:
            shutil.rmtree(self.stage, ignore_errors=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--release-id", required=True)
    result.add_argument("--stage", type=Path, default=DEFAULT_STAGE)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    result.add_argument("--nix", default=str(HOME / ".nix-profile/bin/nix"))
    result.add_argument("--signing-key", type=Path, default=HOME / ".config/nix/gpu-platform-cache.sec")
    result.add_argument(
        "--cache-uri",
        default="file:///mnt/data/worker/.runtime/nix-cache?compression=zstd&compression-level=1&parallel-compression=true",
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.signing_key.is_file():
        raise FileNotFoundError(args.signing_key)
    if not args.catalog.is_file():
        raise FileNotFoundError(args.catalog)
    ComponentBuilder(args).build()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
