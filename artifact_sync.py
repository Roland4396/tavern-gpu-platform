#!/usr/bin/env python3
"""Converge shared model assets and node-local links from a declarative catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


CHUNK_SIZE = 8 * 1024 * 1024
CIVITAI_DOWNLOAD_HOSTS = {
    "civitai.com",
    "www.civitai.com",
    "civitai.red",
    "www.civitai.red",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("schema_version") or 0) != 1:
        raise RuntimeError("unsupported artifact catalog schema")
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("artifact catalog is empty")
    return {str(name): dict(spec) for name, spec in artifacts.items()}


def expected_sha256(spec: dict[str, Any]) -> str:
    value = str(spec.get("sha256") or "").strip().lower()
    if value and (len(value) != 64 or any(character not in "0123456789abcdef" for character in value)):
        raise RuntimeError("sha256 must contain exactly 64 hexadecimal characters")
    return value


def validate_file(path: Path, spec: dict[str, Any], *, hash_content: bool = True) -> str | None:
    if not path.is_file():
        return "missing"
    expected_size = spec.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        return f"size mismatch: expected {int(expected_size)}, got {path.stat().st_size}"
    checksum = expected_sha256(spec)
    if checksum and hash_content:
        actual = sha256_file(path)
        if actual != checksum:
            return f"sha256 mismatch: expected {checksum}, got {actual}"
    return None


def resolve_token(source: dict[str, Any]) -> str:
    environment_name = str(source.get("token_env") or "").strip()
    if environment_name:
        value = os.environ.get(environment_name, "").strip()
        if value:
            return value
    token_file = str(source.get("token_file") or "").strip()
    if token_file:
        path = Path(token_file).expanduser()
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return ""


def source_url(source: dict[str, Any]) -> str:
    url = str(source.get("url") or "").strip()
    if url:
        return url
    if str(source.get("type") or "") == "civitai":
        version_id = int(source.get("model_version_id") or 0)
        if version_id > 0:
            return f"https://civitai.com/api/download/models/{version_id}"
    raise RuntimeError("artifact source has no download URL")


def build_download_request(url: str, token: str, offset: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": "vistiloria-gpu-artifact-sync/1"}
    if token:
        parsed = urllib.parse.urlsplit(url)
        if (parsed.hostname or "").lower() in CIVITAI_DOWNLOAD_HOSTS:
            query = [
                (key, value)
                for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                if key != "token"
            ]
            query.append(("token", token))
            url = urllib.parse.urlunsplit(
                parsed._replace(query=urllib.parse.urlencode(query))
            )
        else:
            headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def download_once(url: str, partial: Path, token: str, timeout: float) -> None:
    offset = partial.stat().st_size if partial.is_file() else 0
    request = build_download_request(url, token, offset)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        status = int(getattr(response, "status", None) or 200)
        append = bool(offset and status == 206)
        mode = "ab" if append else "wb"
        with partial.open(mode) as target:
            while chunk := response.read(CHUNK_SIZE):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())


def import_existing_artifact(name: str, target: Path, spec: dict[str, Any]) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    for raw_source in spec.get("import_paths") or []:
        source = Path(str(raw_source))
        if source.is_symlink() or validate_file(source, spec) is not None:
            continue
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".import", dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination, source.open("rb") as existing:
                while chunk := existing.read(CHUNK_SIZE):
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            error = validate_file(temporary, spec)
            if error is not None:
                raise RuntimeError(f"{name} imported artifact failed verification: {error}")
            os.replace(temporary, target)
            return True
        finally:
            temporary.unlink(missing_ok=True)
    return False


def download_artifact(name: str, target: Path, spec: dict[str, Any]) -> None:
    if import_existing_artifact(name, target, spec):
        return
    source = spec.get("source")
    if not isinstance(source, dict):
        raise RuntimeError(f"{name} is missing and has no managed source")
    url = source_url(source)
    token = resolve_token(source)
    if source.get("token_required") is True and not token:
        location = str(source.get("token_file") or source.get("token_env") or "configured secret")
        raise RuntimeError(f"{name} requires a download token at {location}")

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f".{target.name}.part")
    if validate_file(partial, spec) is None:
        os.replace(partial, target)
        return
    expected_size = spec.get("size_bytes")
    if partial.is_file() and expected_size is not None and partial.stat().st_size >= int(expected_size):
        partial.unlink()

    retries = max(1, min(10, int(source.get("retries") or 5)))
    timeout = max(10.0, min(3600.0, float(source.get("timeout_seconds") or 120.0)))
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            download_once(url, partial, token, timeout)
            error = validate_file(partial, spec)
            if error is not None:
                raise RuntimeError(f"downloaded artifact failed verification: {error}")
            os.replace(partial, target)
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {401, 403}:
                raise RuntimeError(f"{name} download authorization failed (HTTP {exc.code})") from exc
        except (OSError, RuntimeError, urllib.error.URLError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(30, 2 ** (attempt - 1)))
    raise RuntimeError(f"{name} download failed after {retries} attempts: {last_error}")


def link_matches(link: Path, target: Path) -> bool:
    return link.is_symlink() and link.resolve(strict=False) == target.resolve(strict=False)


def ensure_link(link: Path, target: Path, spec: dict[str, Any], *, replace_valid_file: bool = False) -> bool:
    if link_matches(link, target):
        return False
    if link.exists() and not link.is_symlink():
        if not replace_valid_file or validate_file(link, spec) is not None:
            raise RuntimeError(f"refusing to replace non-symlink asset path: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.parent / f".{link.name}.{socket.gethostname()}.{os.getpid()}"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)
    return True


def inspect_artifact(name: str, spec: dict[str, Any], *, check: bool) -> dict[str, Any]:
    target = Path(str(spec.get("path") or ""))
    if not target.is_absolute():
        raise RuntimeError(f"{name} path must be absolute")
    changed = False
    file_error = validate_file(target, spec)
    if file_error is not None and not check:
        download_artifact(name, target, spec)
        changed = True
        file_error = validate_file(target, spec)
    link_errors: list[str] = []
    import_paths = {str(Path(str(path))) for path in spec.get("import_paths") or []}
    for raw_link in spec.get("links") or []:
        link = Path(str(raw_link))
        if not link.is_absolute():
            raise RuntimeError(f"{name} link must be absolute: {link}")
        if link_matches(link, target):
            continue
        if check:
            link_errors.append(str(link))
        else:
            changed = ensure_link(link, target, spec, replace_valid_file=str(link) in import_paths) or changed
    state = "ok" if file_error is None and not link_errors else "drift"
    return {
        "artifact": name,
        "state": state,
        "changed": changed,
        "path": str(target),
        "file_error": file_error,
        "link_errors": link_errors,
    }


def run(catalog_path: Path, selected: list[str] | None, *, check: bool) -> tuple[list[dict[str, Any]], int]:
    catalog = load_catalog(catalog_path)
    names = selected or sorted(catalog)
    unknown = sorted(set(names) - set(catalog))
    if unknown:
        raise RuntimeError(f"unknown artifact(s): {', '.join(unknown)}")
    results: list[dict[str, Any]] = []
    failed = False
    for name in names:
        try:
            result = inspect_artifact(name, catalog[name], check=check)
        except Exception as exc:
            result = {"artifact": name, "state": "error", "changed": False, "error": str(exc)}
        failed = failed or result["state"] != "ok"
        results.append(result)
    return results, 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, default=Path(__file__).with_name("artifacts.json"))
    parser.add_argument("--artifact", action="append", dest="artifacts")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    results, exit_code = run(args.catalog, args.artifacts, check=args.check)
    print(json.dumps({"status": "ok" if exit_code == 0 else "drift", "check": args.check, "results": results}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
