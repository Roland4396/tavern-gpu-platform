#!/usr/bin/env python3
"""Create content manifests for model/checkpoint assets.

The command never imports a model.  Metadata-only mode is quick; `--hash`
explicitly opts into reading every weight file and is intended for release
promotion on the designated builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def files_for(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file())


def build_manifest(asset_id: str, paths: list[Path], *, hash_content: bool) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for root in paths:
        root = root.resolve()
        for path in files_for(root):
            stat = path.stat()
            relative = path.name if root.is_file() else str(path.relative_to(root))
            entry: dict[str, Any] = {"path": str(root), "relative": relative, "size": stat.st_size}
            if hash_content:
                entry["sha256"] = sha256_file(path)
            entries.append(entry)
    digest_input = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "schema_version": 1,
        "asset_id": asset_id,
        "created_at": int(time.time()),
        "builder": platform.node(),
        "content_hashed": hash_content,
        "entries": entries,
        "tree_sha256": hashlib.sha256(digest_input).hexdigest(),
    }


def load_assets(stack_path: Path) -> dict[str, list[Path]]:
    stack = json.loads(stack_path.read_text(encoding="utf-8"))
    assets = {}
    for asset_id, spec in dict(stack.get("artifacts") or {}).items():
        raw = spec.get("paths") or ([spec.get("path")] if spec.get("path") else [])
        assets[asset_id] = [Path(str(path)) for path in raw]
    return assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=Path, default=Path(__file__).with_name("stack.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", action="append", dest="artifacts")
    parser.add_argument("--hash", action="store_true", dest="hash_content")
    args = parser.parse_args(argv)
    assets = load_assets(args.stack)
    selected = args.artifacts or sorted(assets)
    unknown = sorted(set(selected) - set(assets))
    if unknown:
        parser.error(f"unknown artifact(s): {', '.join(unknown)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for asset_id in selected:
        manifest = build_manifest(asset_id, assets[asset_id], hash_content=args.hash_content)
        target = args.output_dir / f"{asset_id}.json"
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        print(json.dumps({"asset_id": asset_id, "tree_sha256": manifest["tree_sha256"], "entries": len(manifest["entries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
