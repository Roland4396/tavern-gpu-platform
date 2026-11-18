#!/usr/bin/env python3
"""Compare node inventories and report environment drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _index(items: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(item.get(key)): item for item in items}


def _compare_map(
    left: dict[str, Any],
    right: dict[str, Any],
    section: str,
    key: str,
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    result = []
    li = _index(left.get(section) or [], key)
    ri = _index(right.get(section) or [], key)
    for name in sorted(set(li) | set(ri)):
        if name not in li or name not in ri:
            result.append({"section": section, "item": name, "kind": "missing"})
            continue
        for field in fields:
            if li[name].get(field) != ri[name].get(field):
                result.append(
                    {
                        "section": section,
                        "item": name,
                        "field": field,
                        "left": li[name].get(field),
                        "right": ri[name].get(field),
                    }
                )
    return result


def compare(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    drift: list[dict[str, Any]] = []
    for field in ("system", "gpu_count", "gpu_model", "driver_version"):
        if left.get("host", {}).get(field) != right.get("host", {}).get(field):
            drift.append(
                {
                    "section": "host",
                    "field": field,
                    "left": left.get("host", {}).get(field),
                    "right": right.get("host", {}).get(field),
                }
            )
    if left.get("workloads") != right.get("workloads"):
        drift.append({"section": "workloads", "left": left.get("workloads"), "right": right.get("workloads")})
    drift.extend(
        _compare_map(
            left,
            right,
            "python_environments",
            "python",
            ("version", "pip_freeze_sha256", "pip_freeze_lines"),
        )
    )
    drift.extend(
        _compare_map(
            left,
            right,
            "git_repositories",
            "path",
            ("head", "status_sha256", "tracked_diff_sha256"),
        )
    )
    drift.extend(_compare_map(left, right, "graph_custom_nodes", "name", ("head", "git_root")))
    drift.extend(
        _compare_map(
            left,
            right,
            "source_trees",
            "path",
            ("kind", "head", "status_sha256", "tracked_diff_sha256", "tree_sha256", "file_count"),
        )
    )
    drift.extend(
        _compare_map(
            left,
            right,
            "runtime_assets",
            "id",
            (
                "kind",
                "path",
                "sha256",
                "size",
                "pip_freeze_sha256",
                "head",
                "status_sha256",
                "tracked_diff_sha256",
                "metadata_sha256",
            ),
        )
    )
    return drift


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args(argv)
    left = json.loads(args.left.read_text(encoding="utf-8"))
    right = json.loads(args.right.read_text(encoding="utf-8"))
    drift = compare(left, right)
    print(json.dumps({"drift_count": len(drift), "drift": drift}, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
