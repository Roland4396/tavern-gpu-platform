from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from artifact_sync import build_download_request, run


class ArtifactSyncTests(unittest.TestCase):
    def test_civitai_token_uses_query_parameter_across_storage_redirect(self) -> None:
        request = build_download_request(
            "https://civitai.red/api/download/models/123?type=Model",
            "secret-token",
            4096,
        )

        query = parse_qs(urlsplit(request.full_url).query)
        self.assertEqual(query["token"], ["secret-token"])
        self.assertEqual(query["type"], ["Model"])
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.get_header("Range"), "bytes=4096-")

    def test_non_civitai_token_uses_bearer_header(self) -> None:
        request = build_download_request(
            "https://models.example/download/model.bin", "secret-token"
        )

        self.assertEqual(request.full_url, "https://models.example/download/model.bin")
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-token")

    def write_catalog(self, root: Path, source: Path, target: Path, link: Path) -> Path:
        content = source.read_bytes()
        catalog = root / "artifacts.json"
        catalog.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifacts": {
                        "fixture": {
                            "path": str(target),
                            "size_bytes": len(content),
                            "sha256": hashlib.sha256(content).hexdigest(),
                            "source": {"url": source.as_uri(), "retries": 1},
                            "links": [str(link)],
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return catalog

    def test_sync_downloads_verifies_and_links_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            source.write_bytes(b"managed model fixture")
            target = root / "shared" / "model.bin"
            link = root / "node" / "models" / "model.bin"
            catalog = self.write_catalog(root, source, target, link)

            check_results, check_code = run(catalog, None, check=True)
            self.assertEqual(check_code, 1)
            self.assertEqual(check_results[0]["state"], "drift")

            sync_results, sync_code = run(catalog, None, check=False)
            self.assertEqual(sync_code, 0)
            self.assertTrue(sync_results[0]["changed"])
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(link.resolve(), target)

            second_results, second_code = run(catalog, None, check=False)
            self.assertEqual(second_code, 0)
            self.assertFalse(second_results[0]["changed"])

    def test_bad_download_never_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.bin"
            source.write_bytes(b"wrong content")
            target = root / "shared" / "model.bin"
            link = root / "node" / "model.bin"
            catalog = self.write_catalog(root, source, target, link)
            value = json.loads(catalog.read_text(encoding="utf-8"))
            value["artifacts"]["fixture"]["sha256"] = hashlib.sha256(b"expected").hexdigest()
            catalog.write_text(json.dumps(value), encoding="utf-8")

            results, exit_code = run(catalog, None, check=False)

            self.assertEqual(exit_code, 1)
            self.assertEqual(results[0]["state"], "error")
            self.assertFalse(target.exists())
            self.assertFalse(link.exists())

    def test_sync_imports_and_replaces_only_declared_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            existing = root / "node" / "models" / "model.bin"
            existing.parent.mkdir(parents=True)
            existing.write_bytes(b"existing managed model")
            target = root / "shared" / "model.bin"
            catalog = self.write_catalog(root, existing, target, existing)
            value = json.loads(catalog.read_text(encoding="utf-8"))
            spec = value["artifacts"]["fixture"]
            spec["source"] = {"url": "https://invalid.example/model.bin", "retries": 1}
            spec["import_paths"] = [str(existing)]
            catalog.write_text(json.dumps(value), encoding="utf-8")

            results, exit_code = run(catalog, None, check=False)

            self.assertEqual(exit_code, 0)
            self.assertEqual(results[0]["state"], "ok")
            self.assertTrue(existing.is_symlink())
            self.assertEqual(existing.resolve(), target)
            self.assertEqual(target.read_bytes(), b"existing managed model")


if __name__ == "__main__":
    unittest.main()
