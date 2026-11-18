#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from h3_release_guard import load_manifest, sha256_file, verify_release


DEFAULT_RELEASE_ROOT = Path("/home/worker/.local/releases/h3")
DEFAULT_FL2VA_LORA = Path(
    "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/loras/"
    "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
)
DEFAULT_REF2VA_LORA = Path(
    "/mnt/data/worker/.runtime/artifacts/visual/h3-sglang/MiniMax-H3/loras/"
    "minimax_h3_turbo_4step_ema_ckpt850.safetensors"
)
CRITICAL_SOURCE_FILES = (
    "python/sglang/multimodal_gen/configs/models/dits/minimax_h3.py",
    "python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py",
    "python/sglang/multimodal_gen/runtime/distributed/parallel_state.py",
    "python/sglang/multimodal_gen/runtime/entrypoints/http_server.py",
    "python/sglang/multimodal_gen/runtime/entrypoints/openai/video_api.py",
    "python/sglang/multimodal_gen/runtime/layers/attention/backends/sage_attn.py",
    "python/sglang/multimodal_gen/runtime/layers/lora/linear.py",
    "python/sglang/multimodal_gen/runtime/loader/h3_rank_sharded_checkpoint.py",
    "python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py",
    "python/sglang/multimodal_gen/runtime/pipelines_core/lora_pipeline.py",
)
RUNTIME_FILES = ("h3_launcher.py", "h3_release_guard.py")
RELEASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def command_output(command: list[str], **kwargs: Any) -> str:
    return run(command, stdout=subprocess.PIPE, **kwargs).stdout.strip()


def validate_release_id(release_id: str) -> None:
    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(f"invalid release id: {release_id!r}")


def replace_path_references(venv: Path, old_paths: tuple[str, ...], release: Path) -> None:
    candidates: set[Path] = {venv / "pyvenv.cfg"}
    candidates.update(path for path in (venv / "bin").iterdir() if path.is_file())
    site_packages = venv / "lib/python3.12/site-packages"
    candidates.update(site_packages.glob("*.pth"))
    candidates.update(site_packages.glob("__editable__*.py"))
    candidates.update(site_packages.glob("*.dist-info/direct_url.json"))
    replacements = {
        old_paths[0]: str(release / "venv"),
        old_paths[1]: str(release / "source"),
    }
    for path in candidates:
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def copy_tree(source: Path, destination: Path, excludes: tuple[str, ...] = ()) -> None:
    command = ["rsync", "-a", "--delete"]
    for value in excludes:
        command.extend(["--exclude", value])
    command.extend([f"{source}/", f"{destination}/"])
    destination.mkdir(parents=True, exist_ok=True)
    run(command)


def git_metadata(source: Path) -> tuple[str, str]:
    commit = command_output(["git", "-C", str(source), "rev-parse", "HEAD"])
    diff = subprocess.run(
        ["git", "-C", str(source), "diff", "--binary", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    untracked = command_output(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"]
    ).encode()
    digest = hashlib.sha256(diff + b"\0" + untracked).hexdigest()
    return commit, digest


def critical_hashes(release: Path) -> dict[str, str]:
    relative_paths = [f"source/{path}" for path in CRITICAL_SOURCE_FILES]
    relative_paths.extend(f"runtime/{path}" for path in RUNTIME_FILES)
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        target = release / relative
        if not target.is_file():
            raise FileNotFoundError(f"critical H3 release file is missing: {target}")
        hashes[relative] = sha256_file(target)
    return hashes


def build(args: argparse.Namespace) -> None:
    validate_release_id(args.release_id)
    release_root = args.release_root.resolve()
    release = release_root / args.release_id
    if release.exists() or release.is_symlink():
        raise FileExistsError(f"release already exists: {release}")
    for path in (args.venv, args.source, args.runtime, args.fl2va_lora, args.ref2va_lora):
        if not path.exists():
            raise FileNotFoundError(path)

    release.mkdir(parents=True)
    try:
        copy_tree(args.venv.resolve(), release / "venv", ("__pycache__", "*.pyc"))
        copy_tree(
            args.source.resolve(),
            release / "source",
            (".git", "__pycache__", "*.pyc"),
        )
        runtime = release / "runtime"
        runtime.mkdir()
        for filename in RUNTIME_FILES:
            source = args.runtime / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, runtime / filename)

        replace_path_references(
            release / "venv",
            (str(args.venv.resolve()), str(args.source.resolve())),
            release,
        )
        python = release / "venv/bin/python"
        source_commit, source_patch_sha256 = git_metadata(args.source)
        python_version = command_output([str(python), "-VV"])
        freeze = command_output([str(python), "-m", "pip", "freeze"])
        import_probe = json.loads(
            command_output(
                [
                    str(python),
                    "-c",
                    (
                        "import json,sageattention,sglang;"
                        "print(json.dumps({'sglang':sglang.__file__,"
                        "'sageattention':sageattention.__file__}))"
                    ),
                ],
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        )
        if not str(import_probe["sglang"]).startswith(str(release / "source")):
            raise RuntimeError(f"release imports SGLang outside itself: {import_probe}")
        if not str(import_probe["sageattention"]).startswith(str(release / "venv")):
            raise RuntimeError(f"release imports SageAttention outside itself: {import_probe}")

        manifest = {
            "schema_version": 1,
            "release_id": args.release_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "builder_node": args.builder_node,
            "source_commit": source_commit,
            "source_patch_sha256": source_patch_sha256,
            "python_version": python_version,
            "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
            "model_contracts": {
                "fl2va": {
                    "lora_sha256": sha256_file(args.fl2va_lora),
                    "expected_lora_covered_layers": 208,
                    "expected_lora_total_layers": 266,
                },
                "ref2va": {
                    "lora_sha256": sha256_file(args.ref2va_lora),
                    "expected_lora_covered_layers": 208,
                    "expected_lora_total_layers": 266,
                },
            },
            "critical_files": {},
        }
        manifest["critical_files"] = critical_hashes(release)
        (release / "release-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_release(release / "release-manifest.json", python)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = args.output_dir / f".{args.release_id}.{uuid.uuid4().hex}.tar.zst"
        run(
            [
                "tar",
                "--sort=name",
                "--mtime=UTC 2020-01-01",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "-I",
                "zstd -T0 -10",
                "-cf",
                str(temporary),
                args.release_id,
            ],
            cwd=release_root,
        )
        artifact_sha256 = sha256_file(temporary)
        artifact = args.output_dir / f"h3-{artifact_sha256}.tar.zst"
        os.replace(temporary, artifact)
        metadata = {
            "release_id": args.release_id,
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha256,
        }
        (args.output_dir / f"h3-{artifact_sha256}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(metadata, sort_keys=True), flush=True)
    except BaseException:
        shutil.rmtree(release, ignore_errors=True)
        raise


def verify_installed(release: Path, runtime_import: bool = True) -> dict[str, Any]:
    manifest_path = release / "release-manifest.json"
    manifest = verify_release(manifest_path, release / "venv/bin/python")
    raw = load_manifest(manifest_path)
    freeze = command_output([str(release / "venv/bin/python"), "-m", "pip", "freeze"])
    actual_freeze = hashlib.sha256(freeze.encode()).hexdigest()
    if actual_freeze != raw.get("pip_freeze_sha256"):
        raise RuntimeError(
            f"H3 pip environment mismatch: expected {raw.get('pip_freeze_sha256')}, "
            f"got {actual_freeze}"
        )
    if runtime_import:
        output = json.loads(
            command_output(
                [
                    str(release / "venv/bin/python"),
                    "-c",
                    (
                        "import json,sageattention,sglang;"
                        "print(json.dumps({'sglang':sglang.__file__,"
                        "'sageattention':sageattention.__file__}))"
                    ),
                ],
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        )
        for key, directory in (("sglang", "source"), ("sageattention", "venv")):
            if not str(output[key]).startswith(str(release / directory)):
                raise RuntimeError(f"H3 {key} imports outside release: {output[key]}")
        manifest["imports"] = output
    return manifest


def safe_archive_members(artifact: Path, release_id: str) -> None:
    listing = command_output(["tar", "--zstd", "-tf", str(artifact)])
    prefix = f"{release_id}/"
    for member in listing.splitlines():
        normalized = member.removeprefix("./")
        if normalized == release_id:
            continue
        if not normalized.startswith(prefix) or ".." in Path(normalized).parts:
            raise RuntimeError(f"unsafe H3 release archive member: {member}")


def atomic_symlink(path: Path, target: Path) -> None:
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}"
    temporary.symlink_to(target)
    os.replace(temporary, path)


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            continue
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(0o555)
        elif stat.S_IXUSR & mode:
            path.chmod(0o555)
        else:
            path.chmod(0o444)
    root.chmod(0o555)


def install(args: argparse.Namespace) -> None:
    validate_release_id(args.release_id)
    actual = sha256_file(args.artifact)
    if actual != args.sha256:
        raise RuntimeError(
            f"H3 release artifact hash mismatch: expected {args.sha256}, got {actual}"
        )
    safe_archive_members(args.artifact, args.release_id)
    root = args.release_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    final = root / args.release_id
    if not final.exists():
        with tempfile.TemporaryDirectory(prefix=".unpack-", dir=root) as temporary:
            staging = Path(temporary)
            run(["tar", "--zstd", "-xf", str(args.artifact), "-C", str(staging)])
            extracted = staging / args.release_id
            verify_installed(extracted)
            make_read_only(extracted)
            os.replace(extracted, final)
    result = verify_installed(final)
    make_read_only(final)

    current = root / "current"
    previous = root / "previous"
    if current.is_symlink():
        old = current.resolve()
        if old != final:
            atomic_symlink(previous, old)
    elif current.exists():
        raise RuntimeError(f"refusing to replace non-symlink H3 current path: {current}")
    atomic_symlink(current, final)
    result["artifact_sha256"] = args.sha256
    print(json.dumps(result, sort_keys=True), flush=True)


def rollback(args: argparse.Namespace) -> None:
    root = args.release_root.resolve()
    current = root / "current"
    previous = root / "previous"
    if not previous.is_symlink():
        raise RuntimeError("H3 previous release is not available")
    target = previous.resolve()
    verify_installed(target)
    old = current.resolve() if current.is_symlink() else None
    atomic_symlink(current, target)
    if old is not None:
        atomic_symlink(previous, old)
    print(json.dumps({"release_id": target.name, "status": "rolled_back"}))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--release-id", required=True)
    build_parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    build_parser.add_argument("--venv", type=Path, required=True)
    build_parser.add_argument("--source", type=Path, required=True)
    build_parser.add_argument("--runtime", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    build_parser.add_argument("--builder-node", default="unknown")
    build_parser.add_argument("--fl2va-lora", type=Path, default=DEFAULT_FL2VA_LORA)
    build_parser.add_argument("--ref2va-lora", type=Path, default=DEFAULT_REF2VA_LORA)
    build_parser.set_defaults(handler=build)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--release-id", required=True)
    install_parser.add_argument("--artifact", type=Path, required=True)
    install_parser.add_argument("--sha256", required=True)
    install_parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    install_parser.set_defaults(handler=install)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("release", type=Path)
    verify_parser.set_defaults(
        handler=lambda args: print(json.dumps(verify_installed(args.release), sort_keys=True))
    )

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    rollback_parser.set_defaults(handler=rollback)
    return result


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
