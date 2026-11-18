#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"invalid H3 release manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported H3 release manifest schema: {path}")
    release_id = manifest.get("release_id")
    if not isinstance(release_id, str) or not release_id:
        raise RuntimeError(f"H3 release manifest has no release_id: {path}")
    return manifest


def verify_release(
    manifest_path: Path,
    python_path: Path,
    *,
    variant: str | None = None,
    lora_path: Path | None = None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    release_root = manifest_path.parent
    manifest = load_manifest(manifest_path)
    if release_root.name != manifest["release_id"]:
        raise RuntimeError(
            f"H3 release directory {release_root.name!r} does not match "
            f"manifest release_id {manifest['release_id']!r}"
        )

    expected_python = (release_root / "venv/bin/python").resolve()
    if python_path.resolve() != expected_python:
        raise RuntimeError(
            f"H3 Python {python_path.resolve()} is outside release {release_root}"
        )

    critical_files = manifest.get("critical_files")
    if not isinstance(critical_files, dict) or not critical_files:
        raise RuntimeError("H3 release manifest has no critical file hashes")
    for relative, expected in critical_files.items():
        target = release_root / str(relative)
        if not target.is_file():
            raise RuntimeError(f"H3 release file is missing: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"H3 release file hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )

    contract: dict[str, Any] | None = None
    if variant:
        contracts = manifest.get("model_contracts") or {}
        contract = contracts.get(variant.lower())
        if not isinstance(contract, dict):
            raise RuntimeError(f"H3 release has no model contract for {variant}")
        if lora_path is not None:
            expected_lora = str(contract.get("lora_sha256") or "")
            if not expected_lora:
                raise RuntimeError(f"H3 release has no LoRA hash for {variant}")
            actual_lora = sha256_file(lora_path)
            if actual_lora != expected_lora:
                raise RuntimeError(
                    f"H3 {variant} LoRA hash mismatch: expected {expected_lora}, "
                    f"got {actual_lora}"
                )

    result = {
        "release_id": manifest["release_id"],
        "source_commit": manifest.get("source_commit"),
    }
    if contract is not None:
        result["expected_lora_covered_layers"] = contract.get(
            "expected_lora_covered_layers"
        )
        result["expected_lora_total_layers"] = contract.get(
            "expected_lora_total_layers"
        )
    return result


def verify_from_environment() -> dict[str, Any] | None:
    raw_manifest = os.environ.get("H3_RELEASE_MANIFEST", "").strip()
    if not raw_manifest:
        return None
    raw_lora = os.environ.get("H3_LORA_PATH", "").strip()
    return verify_release(
        Path(raw_manifest),
        Path(os.environ["H3_PYTHON"]),
        variant=os.environ.get("H3_MODEL_VARIANT", "FL2VA"),
        lora_path=Path(raw_lora) if raw_lora else None,
    )
