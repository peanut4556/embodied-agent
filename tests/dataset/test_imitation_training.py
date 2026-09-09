"""Guard against accidental test-set use during training and checkpoint round-trip."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from embodied_agent.demonstrations import IMAGE_KEY, JOINTS, TASK
from embodied_agent.imitation import ContextPolicy, train


class TrainingTests(unittest.TestCase):
    def test_training_never_reads_test_frames_and_checkpoint_predicts(self):
        accessed = []
        centers = [120, 130, 140, 150, 160, 145, 135]

        def frame(index):
            if index >= 24:  # Last four frames are the reserved test episode.
                raise AssertionError("test data was read during training")
            accessed.append(index)
            image = np.zeros((3, 240, 320), dtype=np.float32)
            center = centers[index // 4]
            image[:, 114:126, center - 6 : center + 6] = np.array([0.9, 0.1, 0.1])[:, None, None]
            action = np.array([-1 + center / 1000, 1.0, 0.1, 0.03, 0.03], dtype=np.float32)
            return {
                IMAGE_KEY: SimpleNamespace(numpy=lambda: image),
                "action": SimpleNamespace(numpy=lambda: action),
            }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, model = root / "dataset", root / "model"
            data.mkdir()
            manifest = {
                "status": "validated",
                "joint_names": JOINTS,
                "task": TASK,
                "episodes": [{"initial_cube_x": 0.28 + i * 0.01, "frames": 4} for i in range(7)],
                "repo_id": "local/test",
                "fps": 25,
                "joint_units": ["rad"] * 3 + ["m"] * 2,
                "model_sha256": "test-model",
                "camera": {},
            }
            (data / "recording.json").write_text(json.dumps(manifest))
            (data / "validation.json").write_text('{"success": true}')
            split = root / "split.json"
            split.write_text(json.dumps({"train": [0, 1, 2, 3, 4], "validation": [5], "test": [6]}))
            with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset") as dataset:
                dataset.return_value.__getitem__.side_effect = frame
                train(data, model, split)
            self.assertEqual(set(accessed), set(range(24)))
            policy = ContextPolicy(model)
            rgb = np.zeros((240, 320, 3), dtype=np.uint8)
            rgb[114:126, 134:146] = [220, 30, 40]
            np.testing.assert_allclose(
                policy.plan(rgb)[0], [-0.86, 1.0, 0.1, 0.03, 0.03], atol=1e-4
            )
            rgb[:] = 0
            rgb[114:126, 184:196] = [220, 30, 40]
            with self.assertRaisesRegex(ValueError, "outside the training range"):
                policy.plan(rgb)
