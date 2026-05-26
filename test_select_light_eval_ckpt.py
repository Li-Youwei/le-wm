from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class SelectLightEvalCkptTest(unittest.TestCase):
    def test_can_select_from_one_suite_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ckpt = tmp_path / "lewm_step_100_object.ckpt"
            ckpt.touch()
            candidates = tmp_path / "candidates.json"
            candidates.write_text(
                json.dumps(
                    {
                        "all": [
                            {
                                "ckpt": str(ckpt),
                                "step": 100,
                                "value": 0.25,
                            }
                        ]
                    }
                )
            )
            (tmp_path / "light_eval_lewm_step_100_object_libero_spatial.log").write_text(
                "  Task  0:  3/5 ( 60.0%) - example\n"
                "  Task  1:  4/5 ( 80.0%) - example\n"
            )
            out_json = tmp_path / "selected.json"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "select_light_eval_ckpt.py"),
                    "--ckpt-dir",
                    str(tmp_path),
                    "--candidates-json",
                    str(candidates),
                    "--suites",
                    "libero_spatial",
                    "--out",
                    str(out_json),
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(out_json.read_text())
            self.assertEqual(payload["top_1"], str(ckpt))
            self.assertEqual(payload["top_1_successes"], 7)
            self.assertEqual(payload["top_1_episodes"], 10)
            self.assertEqual(list(payload["all"][0]["suites"].keys()), ["libero_spatial"])


if __name__ == "__main__":
    unittest.main()
