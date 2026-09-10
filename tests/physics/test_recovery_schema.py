import json
import tempfile
import unittest
from pathlib import Path

from embodied_agent.imitation import dataset_manifest
from embodied_agent.recovery_data import validate_recovery


class RecoverySchemaTests(unittest.TestCase):
    def test_legacy_success_only_trainer_rejects_recovery_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "recording.json").write_text(
                json.dumps({"dataset_type": "feedback_recovery", "status": "validated"})
            )
            with self.assertRaisesRegex(ValueError, "dedicated trainer"):
                dataset_manifest(root)

    def test_partial_recording_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "recording.json").write_text(
                json.dumps({"dataset_type": "feedback_recovery", "status": "failed"})
            )
            with self.assertRaisesRegex(ValueError, "complete recovery"):
                validate_recovery(root)
