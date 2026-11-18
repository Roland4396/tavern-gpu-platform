from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from targeted_release import release_profiles, replace_component


class TargetedReleaseTests(unittest.TestCase):
    def test_replaces_only_requested_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = Path("/nix/store/new-component")
            source = root / "components.json"
            output = root / "next.json"
            source.write_text(
                json.dumps(
                    {
                        "release_id": "old",
                        "components": {
                            "gpu-runtime-source": {"store_path": "/nix/store/old", "cache_published": True},
                            "runtime-core-env": {"store_path": "/nix/store/runtime"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(Path, "is_dir", return_value=True):
                result = replace_component(source, output, "next", "gpu-runtime-source", store)

            self.assertEqual(result["release_id"], "next")
            self.assertEqual(result["components"]["gpu-runtime-source"]["store_path"], str(store))
            self.assertFalse(result["components"]["gpu-runtime-source"]["cache_published"])
            self.assertEqual(
                result["components"]["runtime-core-env"]["store_path"],
                "/nix/store/runtime",
            )

    def test_replaces_requested_profiles_and_preserves_others(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "active.json"
            output = root / "release.json"
            active.write_text(
                json.dumps({"profiles": {"contract": "/nix/store/old-contract", "graph": "/nix/store/graph"}}),
                encoding="utf-8",
            )

            replacement = Path("/nix/store/new-contract")
            with patch.object(Path, "is_dir", return_value=True):
                result = release_profiles(active, output, "next", {"contract": replacement})

            self.assertEqual(result["gpu_stack_release_id"], "next")
            self.assertEqual(result["gpu_stack_profiles"]["contract"], str(replacement))
            self.assertEqual(result["gpu_stack_profiles"]["graph"], "/nix/store/graph")


if __name__ == "__main__":
    unittest.main()
