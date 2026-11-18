#!/usr/bin/env python3
"""Collect a comparable inventory of our GPU runtime on a shared node.

The inventory is intentionally service-agnostic but ownership-bounded. It starts
from node.json and the platform contract's ``owned_components`` and records
only that dependency closure. It never walks other users' home directories,
loads a model, or initializes CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CONFIG = Path.home() / ".config/gpu-runtime/node.json"
SECRET_KEY = re.compile(r"(?:token|secret|password|private[_-]?key|credential)", re.I)
URL_CREDENTIAL = re.compile(r"(https?://)[^/\s@]+:[^@\s]+@", re.I)
MODEL_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


def _run(argv: list[str], *, cwd: Path | None = None, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _redact_freeze_line(line: str) -> str:
    return URL_CREDENTIAL.sub(r"\1<redacted>@", line)


def _redact(value: Any, key: str = "") -> Any:
    if SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, dict):
        return {name: _redact(item, name) for name, item in sorted(value.items())}
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    return value


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _absolute_paths(value: Any) -> set[Path]:
    result: set[Path] = set()
    for item in _walk_strings(value):
        for part in item.split(os.pathsep):
            if part.startswith("/"):
                result.add(Path(part))
    return result


def _environment_root(python: Path) -> Path | None:
    """Return the environment root for a POSIX or Conda interpreter."""
    try:
        resolved = python.resolve()
    except OSError:
        resolved = python
    if resolved.parent.name != "bin":
        return None
    root = resolved.parent.parent
    if (root / "pyvenv.cfg").is_file() or (root / "conda-meta").is_dir():
        return root
    return None


def _discover_python_paths(config: dict[str, Any], owned_components: dict[str, Any]) -> set[Path]:
    candidates: set[Path] = set()
    for path in _absolute_paths(config):
        if path.name.startswith("python") and path.parent.name == "bin":
            candidates.add(path)
        elif (path / "bin/python").exists():
            candidates.add(path / "bin/python")

    for spec in owned_components.values():
        if str(spec.get("kind")) != "python-environment":
            continue
        root = Path(str(spec.get("path") or ""))
        candidate = root / "bin/python"
        if candidate.exists():
            candidates.add(candidate)
    return candidates


def _path_record(path: Path, *, hash_large_files: bool) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        record["kind"] = "missing"
        return record
    record.update({"mode": stat.st_mode & 0o7777, "mtime_ns": stat.st_mtime_ns})
    if path.is_symlink():
        record.update({"kind": "symlink", "target": os.readlink(path)})
    elif path.is_file():
        record.update({"kind": "file", "size": stat.st_size})
        if hash_large_files or stat.st_size <= 64 * 1024 * 1024:
            record["sha256"] = _sha256_file(path)
        else:
            record["sha256"] = None
    elif path.is_dir():
        record["kind"] = "directory"
    else:
        record["kind"] = "other"
    return record


def _python_environments(config: dict[str, Any], owned_components: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = _discover_python_paths(config, owned_components)

    environments = []
    seen_roots: set[Path] = set()
    for python in sorted(candidates):
        if not python.exists():
            continue
        root = _environment_root(python)
        if root is not None:
            root = root.resolve()
            if root in seen_roots:
                continue
            seen_roots.add(root)
        version = _run([str(python), "-c", "import sys; print(sys.version.replace('\\n', ' '))"])
        freeze = _run([str(python), "-m", "pip", "freeze", "--all"], timeout=180)
        freeze_lines = [_redact_freeze_line(line) for line in freeze.splitlines()]
        environments.append(
            {
                "python": str(python),
                "real_python": str(python.resolve()),
                "version": version,
                "pip_freeze_lines": len(freeze_lines),
                "pip_freeze_sha256": _sha256_bytes(freeze.encode()),
                "pip_freeze": freeze_lines,
            }
        )
    return environments


def _git_root(path: Path) -> Path | None:
    candidate = path if path.is_dir() else path.parent
    output = _run(["git", "-C", str(candidate), "rev-parse", "--show-toplevel"])
    return Path(output) if output else None


def _discover_git_roots(config: dict[str, Any], owned_components: dict[str, Any]) -> set[Path]:
    owned_paths = [Path(str(spec.get("path") or "")).resolve() for spec in owned_components.values() if spec.get("path")]
    roots: set[Path] = set()
    for path in _absolute_paths(config):
        resolved = path.resolve() if path.exists() else path
        if not any(resolved == root or root in resolved.parents for root in owned_paths):
            continue
        if root := _git_root(path):
            roots.add(root)
    for spec in owned_components.values():
        if str(spec.get("kind")) not in {"source-tree", "runtime-tree"}:
            continue
        root = _git_root(Path(str(spec.get("path") or "")))
        if root:
            roots.add(root)
    return roots


def _source_roots(config: dict[str, Any], repositories: list[dict[str, Any]], owned_components: dict[str, Any]) -> list[Path]:
    owned_source_paths = {str(Path(str(spec.get("path") or "")).resolve()) for spec in owned_components.values() if str(spec.get("kind")) in {"source-tree", "runtime-tree"}}
    roots = {Path(item["path"]) for item in repositories if item.get("path") and str(Path(item["path"]).resolve()) in owned_source_paths}
    for spec in owned_components.values():
        if str(spec.get("kind")) in {"source-tree", "runtime-tree"}:
            path = Path(str(spec.get("path") or ""))
            if path.exists():
                roots.add(path)
    return sorted(path for path in roots if path.exists())


def _source_tree_record(path: Path, repository: dict[str, Any] | None = None, kind: str = "source-tree") -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "kind": kind}
    if repository:
        record.update({key: repository.get(key) for key in ("head", "status_sha256", "tracked_diff_sha256", "origin")})
        return record
    digest = hashlib.sha256()
    file_count = 0
    try:
        for item in sorted(path.rglob("*")):
            if not item.is_file() or any(part in {"__pycache__", ".git", "models", "outputs", "logs", "node_modules"} for part in item.parts):
                continue
            relative = str(item.relative_to(path)).encode()
            stat = item.stat()
            digest.update(relative + b"\0" + str(stat.st_size).encode() + b"\0")
            if stat.st_size <= 16 * 1024 * 1024:
                digest.update(_sha256_file(item).encode())
            file_count += 1
    except OSError:
        record["kind"] = "missing"
        return record
    record.update({"tree_sha256": digest.hexdigest(), "file_count": file_count})
    return record


def _config_files(config_path: Path, config: dict[str, Any], owned_components: dict[str, Any]) -> list[dict[str, Any]]:
    paths = {config_path}
    for path in _absolute_paths(config):
        if path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".sh"}:
            paths.add(path)
    for spec in owned_components.values():
        if str(spec.get("kind")) != "runtime-config":
            continue
        path = Path(str(spec.get("path") or ""))
        if path.is_file():
            paths.add(path)
    return [_path_record(path, hash_large_files=False) for path in sorted(paths)]


def _runtime_assets(config_path: Path, config: dict[str, Any], owned_components: dict[str, Any], environments: list[dict[str, Any]], source_trees: list[dict[str, Any]], artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    environments_by_root = {
        str(_environment_root(Path(item["python"])).resolve()): item
        for item in environments
        if _environment_root(Path(item["python"])) is not None
    }
    source_by_path = {str(item["path"]): item for item in source_trees}
    for name, spec in sorted(owned_components.items()):
        kind = str(spec.get("kind") or "")
        path = str(spec.get("path") or "")
        if kind == "python-environment":
            environment = environments_by_root.get(str(Path(path).resolve()))
            if environment is None:
                assets.append({"id": f"owned:{name}", "kind": kind, "path": path, "status": "missing"})
            else:
                assets.append({"id": f"owned:{name}", "kind": kind, "path": path, **{key: environment.get(key) for key in ("python", "pip_freeze_sha256", "pip_freeze_lines")}})
        elif kind in {"source-tree", "runtime-tree"}:
            source = source_by_path.get(path)
            assets.append({"id": f"owned:{name}", "kind": kind, "path": path, **({key: source.get(key) for key in ("head", "status_sha256", "tracked_diff_sha256", "tree_sha256", "file_count") if key in source} if source else {"status": "missing"})})
        elif kind == "runtime-config":
            record = _path_record(Path(path), hash_large_files=False)
            assets.append({"id": f"owned:{name}", **record, "kind": kind, "path": path})
    for record in _config_files(config_path, config, owned_components):
        if not any(item.get("kind") == "runtime-config" and item.get("path") == record["path"] for item in assets):
            assets.append({"id": f"config:{record['path']}", **record, "kind": "runtime-config"})
    for artifact in artifacts:
        assets.append({"id": f"model:{artifact['path']}", **artifact, "kind": "model-asset"})
    return sorted(assets, key=lambda item: item["id"])


def _git_repositories(config: dict[str, Any], owned_components: dict[str, Any]) -> list[dict[str, Any]]:
    repositories = []
    for root in sorted(_discover_git_roots(config, owned_components)):
        status = _run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"])
        diff = _run(["git", "-C", str(root), "diff", "--binary", "HEAD"], timeout=120)
        repositories.append(
            {
                "path": str(root),
                "head": _run(["git", "-C", str(root), "rev-parse", "HEAD"]),
                "branch": _run(["git", "-C", str(root), "branch", "--show-current"]),
                "origin": URL_CREDENTIAL.sub(r"\1<redacted>@", _run(["git", "-C", str(root), "remote", "get-url", "origin"])),
                "dirty_count": len(status.splitlines()),
                "status_sha256": _sha256_bytes(status.encode()),
                "tracked_diff_sha256": _sha256_bytes(diff.encode()),
            }
        )
    return repositories


def _graph_nodes() -> list[dict[str, Any]]:
    root = Path.home() / ".local/share/graph-stack/core/custom_nodes"
    if not root.is_dir():
        return []
    nodes = []
    for path in sorted(
        item for item in root.iterdir() if item.is_dir() and item.name != "__pycache__"
    ):
        git_root = _git_root(path)
        nodes.append(
            {
                "name": path.name,
                "path": str(path),
                "git_root": str(git_root) if git_root else None,
                "head": _run(["git", "-C", str(git_root), "rev-parse", "HEAD"]) if git_root else None,
            }
        )
    return nodes


def _model_artifacts(config: dict[str, Any], *, hash_models: bool, stack_path: Path | None = None) -> list[dict[str, Any]]:
    artifacts = []
    candidate_paths = set(_absolute_paths(config))
    if stack_path and stack_path.is_file():
        try:
            stack = json.loads(stack_path.read_text(encoding="utf-8"))
            for spec in dict(stack.get("artifacts") or {}).values():
                candidate_paths.update(Path(str(item)) for item in (spec.get("paths") or ([spec.get("path")] if spec.get("path") else [])))
        except (OSError, ValueError, TypeError):
            pass
    for path in sorted(candidate_paths):
        name = str(path).lower()
        if not any(token in name for token in ("model", "artifact", "checkpoint", "lora", "shard")):
            continue
        record = _path_record(path, hash_large_files=hash_models)
        if path.is_dir():
            files = []
            for child in sorted(item for item in path.rglob("*") if item.is_file()):
                if child.suffix.lower() not in MODEL_SUFFIXES and child.name not in {
                    "config.json",
                    "model_index.json",
                }:
                    continue
                child_record = _path_record(child, hash_large_files=hash_models)
                child_record["path"] = str(child.relative_to(path))
                files.append(child_record)
            record["files"] = files
            metadata = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
            record["metadata_sha256"] = _sha256_bytes(metadata)
        artifacts.append(record)
    return artifacts


def collect(config_path: Path, *, hash_models: bool = False, stack_path: Path | None = None) -> dict[str, Any]:
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    owned_components: dict[str, Any] = {}
    if stack_path and stack_path.is_file():
        try:
            owned_components = dict(json.loads(stack_path.read_text(encoding="utf-8")).get("owned_components") or {})
        except (OSError, ValueError, TypeError):
            owned_components = {}
    runtime_paths = [_path_record(path, hash_large_files=False) for path in sorted(_absolute_paths(raw_config))]
    driver = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_rows = [row.split(",", 4) for row in driver.splitlines() if row.strip()]
    gpu_names = sorted({row[1].strip() for row in gpu_rows if len(row) > 1})
    driver_versions = sorted({row[3].strip() for row in gpu_rows if len(row) > 3})
    repositories = _git_repositories(raw_config, owned_components)
    source_roots = _source_roots(raw_config, repositories, owned_components)
    repository_by_path = {item["path"]: item for item in repositories}
    source_trees = []
    for path in source_roots:
        kind = next((str(spec.get("kind")) for spec in owned_components.values() if str(spec.get("path")) == str(path)), "source-tree")
        source_trees.append(_source_tree_record(path, repository_by_path.get(str(path)), kind))
    inventory = {
        "schema_version": 1,
        "node_name": raw_config.get("node_name"),
        "host": {
            "hostname": platform.node(),
            "system": platform.system(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "glibc": platform.libc_ver(),
            "nvidia_smi": driver.splitlines(),
            "gpu_count": len(gpu_rows),
            "gpu_model": gpu_names[0] if len(gpu_names) == 1 else gpu_names,
            "driver_version": driver_versions[0] if len(driver_versions) == 1 else driver_versions,
            "cuda_compiler": _run(["nvcc", "--version"]),
        },
        "config": {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "redacted": _redact(raw_config),
        },
        "workloads": sorted((raw_config.get("workloads") or {}).keys()),
        "python_environments": _python_environments(raw_config, owned_components),
        "git_repositories": repositories,
        "source_trees": source_trees,
        "runtime_paths": runtime_paths,
        "graph_custom_nodes": _graph_nodes(),
        "model_artifacts": _model_artifacts(raw_config, hash_models=hash_models, stack_path=stack_path),
    }
    inventory["runtime_assets"] = _runtime_assets(
        config_path,
        raw_config,
        owned_components,
        inventory["python_environments"],
        inventory["source_trees"],
        inventory["model_artifacts"],
    )
    canonical = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    inventory["inventory_sha256"] = _sha256_bytes(canonical)
    return inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stack", type=Path, default=Path(__file__).with_name("stack.json"))
    parser.add_argument(
        "--hash-models",
        action="store_true",
        help="hash large model files; the default records only size and metadata",
    )
    args = parser.parse_args(argv)
    value = collect(args.config, hash_models=args.hash_models, stack_path=args.stack)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.output)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
