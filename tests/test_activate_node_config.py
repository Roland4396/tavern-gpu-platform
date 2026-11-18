from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from activate_node_config import update_config


class UpdateConfigTests(unittest.TestCase):
    def test_gpu_graph_removes_cpu_flag_without_losing_other_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stack = root / "stack.json"
            stack.write_text(json.dumps({"workloads": {
                "graph": {"profile_command": "gpu-run-graph", "gpu_count": 1}
            }}))
            executable = root / "graph/current/bin/gpu-run-graph"
            executable.parent.mkdir(parents=True)
            executable.touch()
            original = {"workloads": {"graph": {
                "gpu_count": 0,
                "command": [str(executable), "--cpu", "--output-directory", "/private/output"],
            }}}
            result = update_config(original, stack, root)
            self.assertEqual(result["workloads"]["graph"]["gpu_count"], 1)
            self.assertEqual(result["workloads"]["graph"]["command"],
                             [str(executable), "--output-directory", "/private/output"])
            self.assertEqual(update_config(result, stack, root), result)

    def test_graph_command_is_idempotent_after_profile_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stack_path = root / "stack.json"
            stack_path.write_text(
                json.dumps(
                    {
                        "workloads": {
                            "graph": {"profile_command": "graph-runtime", "gpu_count": 0},
                            "h3": {"profile_command": "h3-runtime", "gpu_count": 4},
                        }
                    }
                ),
                encoding="utf-8",
            )
            for workload in ("graph", "h3"):
                executable = root / workload / "current" / "bin" / f"{workload}-runtime"
                executable.parent.mkdir(parents=True)
                executable.touch()

            original = {
                "workloads": {
                    "graph": {
                        "gpu_count": 8,
                        "command": [
                            "/home/worker/.venvs/deep-learning/bin/python",
                            "/home/worker/.local/share/graph-stack/core/main.py",
                            "--disable-auto-launch",
                            "--dont-print-server",
                        ]
                    },
                    "h3": {"gpu_count": 8, "command": ["/legacy/h3", "serve"]},
                }
            }

            first = update_config(original, stack_path, root)
            second = update_config(first, stack_path, root)

            self.assertEqual(first, second)
            self.assertEqual(first["workloads"]["graph"]["gpu_count"], 0)
            self.assertEqual(first["workloads"]["h3"]["gpu_count"], 4)
            self.assertEqual(
                first["workloads"]["graph"]["command"],
                [
                    str(root / "graph" / "current" / "bin" / "graph-runtime"),
                    "--cpu",
                    "--disable-auto-launch",
                    "--dont-print-server",
                ],
            )

    def test_optional_profile_preserves_existing_launcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            stack_path = root / "stack.json"
            stack_path.write_text(
                json.dumps(
                    {
                        "workloads": {
                            "qwen": {
                                "profile_command": "qwen-runtime",
                                "profile_optional": True,
                                "gpu_count": 2,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            original = {
                "workloads": {
                    "qwen": {
                        "gpu_count": 8,
                        "profile_path": "/missing/profile",
                        "command": ["/usr/bin/python3", "/opt/qwen_launcher.py"],
                    }
                }
            }

            updated = update_config(original, stack_path, root)

            self.assertEqual(updated["workloads"]["qwen"]["gpu_count"], 2)
            self.assertEqual(
                updated["workloads"]["qwen"]["command"],
                ["/usr/bin/python3", "/opt/qwen_launcher.py"],
            )
            self.assertNotIn("profile_path", updated["workloads"]["qwen"])


if __name__ == "__main__":
    unittest.main()
