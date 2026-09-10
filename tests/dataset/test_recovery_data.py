"""Real recovery round-trip and corruption checks; requires local weights and rendering."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_agent.recovery_data import record_recovery, validate_recovery

MODEL = Path(os.environ.get("RECOVERY_TEST_MODEL", "outputs/models/context-bc-v1"))


@unittest.skipUnless((MODEL / "policy.npz").is_file(), "requires RECOVERY_TEST_MODEL checkpoint")
class RecoveryDataTests(unittest.TestCase):
    def test_roundtrip_keeps_failures_and_invalidates_corrupted_training_selection(self):
        cases = [
            {
                "name": "recovered",
                "x": 0.34,
                "force_at": 4.88,
                "force_duration": 0.08,
                "force_z": -30,
            },
            {
                "name": "stopped",
                "x": 0.30,
                "force_at": 5.0,
                "force_duration": 0.08,
                "force_z": -30,
                "stop_on_recovery": True,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recovery"
            report = record_recovery(root, MODEL, cases)
            self.assertTrue(report["success"])
            self.assertEqual([e["task_success"] for e in report["episodes"]], [True, False])
            selection = json.loads((root / "selection.json").read_text())
            self.assertEqual(selection["successful_episodes"], [0])
            self.assertEqual(selection["failure_episodes"], [1])
            self.assertEqual(selection["independent_test_episodes"], [])
            with self.assertRaises(FileExistsError):
                record_recovery(root, MODEL, cases)
            parquet = next((root / "data").rglob("*.parquet"))
            original = pq.read_table(parquet)
            for column, replacement in (
                ("next.success", True),
                ("environment.force", [0.0, 0.0, 1.0]),
                ("action", [0.0] * 5),
                ("observation.state", [0.0] * 5),
                ("timestamp", 4.0),
            ):
                with self.subTest(column=column):
                    values = original[column].to_pylist()
                    values[0] = replacement
                    index = original.schema.get_field_index(column)
                    altered = original.set_column(
                        index,
                        original.schema.field(index),
                        pa.array(values, type=original.schema.field(index).type),
                    )
                    pq.write_table(altered, parquet)
                    with self.assertRaises((ValueError, AssertionError)):
                        validate_recovery(root)
                    self.assertFalse(json.loads((root / "selection.json").read_text())["usable"])
                    self.assertFalse(json.loads((root / "validation.json").read_text())["success"])
                    self.assertEqual(
                        json.loads((root / "recording.json").read_text())["status"],
                        "validation_failed",
                    )
            pq.write_table(original, parquet)
            self.assertTrue(validate_recovery(root)["success"])
