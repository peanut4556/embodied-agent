"""Real LeRobot round-trip and corrupted recording checks; requires rendering."""

import json
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from embodied_agent.demonstrations import record_dataset, validate_dataset, validate_options


class DatasetTests(unittest.TestCase):
    def test_invalid_collection_options(self):
        for positions, fps in (([], 25), ([float("nan")], 25), ([0.6], 25), ([0.32], 30)):
            with self.subTest(positions=positions, fps=fps), self.assertRaises(ValueError):
                validate_options(positions, fps)

    def test_roundtrip_replay_and_corrupt_timestamp_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            report = record_dataset(root, positions=[0.32])
            self.assertEqual(report["total_frames"], 261)
            self.assertTrue(report["episodes"][0]["replay_success"])
            self.assertTrue((root / "preview.gif").is_file())
            with self.assertRaises(FileExistsError):
                record_dataset(root, positions=[0.32])
            parquet = next((root / "data").rglob("*.parquet"))
            table = pq.read_table(parquet)
            timestamps = table["timestamp"].to_pylist()
            timestamps[1] = 3.0
            index = table.schema.get_field_index("timestamp")
            altered = table.set_column(
                index,
                table.schema.field(index),
                pa.array(timestamps, type=table.schema.field(index).type),
            )
            pq.write_table(altered, parquet)
            with self.assertRaises((ValueError, AssertionError)):
                validate_dataset(root)
            self.assertFalse(json.loads((root / "validation.json").read_text())["success"])
            self.assertEqual(
                json.loads((root / "recording.json").read_text())["status"], "validation_failed"
            )
            # Valid dimensions and timestamps alone must not qualify a broken action sequence.
            index = table.schema.get_field_index("action")
            altered = table.set_column(
                index,
                table.schema.field(index),
                pa.array([[0.0] * 5] * len(table), type=table.schema.field(index).type),
            )
            pq.write_table(altered, parquet)
            with self.assertRaisesRegex(ValueError, "replay failed"):
                validate_dataset(root)

    def test_incomplete_recording_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "recording.json").write_text('{"status": "failed"}')
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_dataset(root)
