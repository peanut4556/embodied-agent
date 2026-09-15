"""Correction provenance, expert-only targets and real replay corruption guards."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from embodied_agent.correction_data import (
    SOURCES,
    CorrectionExpert,
    check_layout,
    payload_digest,
    training_sequences,
    validate,
)
from embodied_agent.demonstrations import IMAGE_KEY, JOINTS
from embodied_agent.imitation import dataset_manifest


class CorrectionDataTests(unittest.TestCase):
    def test_expert_stop_latches_first_target_and_preserves_fingers(self):
        expert = CorrectionExpert.__new__(CorrectionExpert)
        expert.reason, expert.done = "", False
        expert.command = np.array([0.0, 0.0, 0.0, 0.007, 0.007])
        expert.world = SimpleNamespace(
            data=SimpleNamespace(qpos=np.array([0.1, 0.2, 0.3, 0.01, 0.01])),
            model=SimpleNamespace(actuator_ctrlrange=np.array([[-1.0, 1.0]] * 5)),
        )
        first = expert.stop("user stop")
        expert.world.data.qpos[:] = 0.9
        np.testing.assert_array_equal(expert.stop("user stop"), first)
        np.testing.assert_array_equal(first[3:], [0.007, 0.007])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = {
            "dataset_type": "feedback_correction",
            "status": "validated",
            "fps": 25,
            "joint_names": JOINTS,
            "source_names": SOURCES,
            "repo_id": "local/test",
            "policy_input_keys": [IMAGE_KEY, "observation.state", "observation.holding"],
            "split": {"train": [0, 1], "validation": [2], "test": []},
            "episodes": [
                {
                    "index": i,
                    "frames": 4,
                    "success": i != 1,
                    "takeover": {"tick": 1},
                    "scenario": {"x": 0.30 if i < 2 else 0.34, "takeover_seconds": 0.04},
                }
                for i in range(3)
            ],
        }
        self.save("recording.json", self.manifest)
        self.save("validation.json", {"success": True})
        self.save("selection.json", {"usable": True, "payload_sha256": payload_digest(self.root)})

    def save(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def test_learner_prefix_is_context_but_never_target_and_failures_not_loaded(self):
        accessed = []

        def row(index):
            if index >= 4:
                raise AssertionError("failure or validation frame read from training")
            accessed.append(index)
            source = (0, 1, 1, 2)[index]
            return {
                IMAGE_KEY: torch.zeros(3, 240, 320),
                "observation.state": torch.ones(5),
                "observation.holding": torch.tensor(False),
                "action": torch.ones(5),
                "controller.source": torch.tensor(source),
                "supervision.valid": torch.tensor(source == 1),
            }

        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as dataset:
            dataset.return_value.__getitem__.side_effect = row
            samples = list(training_sequences(self.root, "train"))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(accessed, [0, 1, 2, 3])
        np.testing.assert_array_equal(sample["loss_mask"], [False, True, True, False])
        np.testing.assert_array_equal(sample["targets"][[0, 3]], np.zeros((2, 5)))
        self.assertEqual(
            set(sample["inputs"]), {IMAGE_KEY, "observation.state", "observation.holding"}
        )
        self.assertEqual(sample["inputs"]["observation.holding"].shape, (4, 1))

    def test_changed_payload_and_test_group_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "development-only"):
            list(training_sequences(self.root, "test"))
        (self.root / "data").mkdir()
        (self.root / "data/changed.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "unchanged"):
            list(training_sequences(self.root, "train"))

    def test_position_leakage_and_takeover_mismatch_rejected(self):
        self.manifest["split"] = {"train": [0], "validation": [1, 2], "test": []}
        with self.assertRaisesRegex(ValueError, "crosses"):
            check_layout(self.manifest)
        self.manifest["split"] = {"train": [0, 1], "validation": [2], "test": []}
        self.manifest["episodes"][0]["takeover"]["tick"] = 2
        with self.assertRaisesRegex(ValueError, "takeover"):
            check_layout(self.manifest)

    def test_old_unmasked_training_entry_rejects_corrections(self):
        with self.assertRaisesRegex(ValueError, "dedicated trainer"):
            dataset_manifest(self.root)

    @unittest.skipUnless(
        os.environ.get("CORRECTION_TEST_DATASET"), "requires recorded correction dataset"
    )
    def test_real_takeover_corruption_invalidates_selection_and_can_be_revalidated(self):
        root = self.root / "copy"
        shutil.copytree(os.environ["CORRECTION_TEST_DATASET"], root)
        original = (root / "recording.json").read_text()
        manifest = json.loads(original)
        manifest["episodes"][0]["takeover"]["qpos"][0] += 0.1
        (root / "recording.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "takeover state"):
            validate(root)
        self.assertFalse(json.loads((root / "selection.json").read_text())["usable"])
        self.assertEqual(
            json.loads((root / "recording.json").read_text())["status"], "validation_failed"
        )
        (root / "recording.json").write_text(original)
        self.assertTrue(validate(root)["success"])
        self.assertTrue(json.loads((root / "selection.json").read_text())["usable"])
