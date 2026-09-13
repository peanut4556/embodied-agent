"""Causality, outcome isolation and held-out access guards for temporal data."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from embodied_agent.temporal_data import INPUT_KEYS, TemporalDataset, check_source, prepare


class TemporalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "dataset_type": "feedback_recovery",
            "status": "validated",
            "fps": 25,
            "repo_id": "local/test",
            "phase_names": ["running", "checking_grasp", "returning", "completed", "stopped"],
            "episodes": [
                {"index": i, "frames": 8, "success": i != 1, "scenario": {"x": x}}
                for i, x in enumerate([0.30, 0.30, 0.34, 0.38])
            ],
        }
        self.selection = {
            "usable": True,
            "successful_episodes": [0, 2, 3],
            "failure_episodes": [1],
            "independent_test_episodes": [3],
        }
        self.split = {"train": [0, 1], "validation": [2], "test": [3]}
        self.save("recording.json", self.manifest)
        self.save("selection.json", self.selection)
        self.save("validation.json", {"success": True})
        self.accessed = []
        self.patch = patch("lerobot.datasets.lerobot_dataset.LeRobotDataset")
        dataset = self.patch.start()
        self.addCleanup(self.patch.stop)
        dataset.return_value.__len__.return_value = 32
        dataset.return_value.__getitem__.side_effect = self.frame

    def save(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def frame(self, index):
        if index >= 24:
            raise AssertionError("reserved test frame accessed")
        import torch

        self.accessed.append(index)
        ep, frame = divmod(index, 8)
        return {
            INPUT_KEYS[0]: torch.zeros(3, 240, 320),
            "observation.state": torch.full((5,), float(frame)),
            "observation.holding": torch.tensor(False),
            "action": torch.full((5,), float(frame + 10)),
            "episode_index": torch.tensor(ep),
            "frame_index": torch.tensor(frame),
            "timestamp": torch.tensor(frame / 25),
            "phase": torch.tensor(0 if frame < 7 else (4 if ep == 1 else 3)),
        }

    def build(self):
        index = self.root / "index.json"
        report = prepare(self.root, index, self.split, history=3, horizon=2)
        return index, report

    def test_causal_windows_targets_and_failure_isolation(self):
        index, report = self.build()
        self.assertEqual(report["counts"]["train"], {"imitation": 4, "outcome": 12})
        self.assertEqual(report["counts"]["test"], {"imitation": 0, "outcome": 0})
        dataset = TemporalDataset(index)
        self.accessed.clear()
        sample = dataset[0]
        self.assertEqual(set(sample["inputs"]), set(INPUT_KEYS))
        np.testing.assert_array_equal(sample["inputs"]["observation.state"][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(sample["targets"]["action"][:, 0], [12, 13])
        self.assertEqual(self.accessed, [0, 1, 2, 2, 3])
        self.assertTrue(all(ep == 0 for ep, _ in dataset.refs))
        outcomes = TemporalDataset(index, objective="outcome")
        failure = outcomes[6]
        self.assertEqual(failure["episode"], 1)
        self.assertNotIn("action", failure["targets"])
        self.assertFalse(failure["targets"]["episode_success"].item())
        last = dataset[-1]
        self.assertEqual(last["frame"], 5)  # No target at completed frame 7.
        with self.assertRaises(ValueError):
            TemporalDataset(index, group="test")

    def test_same_position_cannot_cross_split_even_with_different_names(self):
        with self.assertRaisesRegex(ValueError, "same initial position"):
            check_source(self.root, {"train": [0], "validation": [1, 2], "test": [3]})

    def test_inspected_samples_cannot_be_promoted_to_test(self):
        self.selection["independent_test_episodes"] = []
        self.save("selection.json", self.selection)
        with self.assertRaisesRegex(ValueError, "previously inspected"):
            self.build()

    def test_duplicate_or_missing_split_is_rejected(self):
        for split in (
            {"train": [0, 1], "validation": [2], "test": [2]},
            {"train": [0], "validation": [2], "test": [3]},
        ):
            with self.assertRaisesRegex(ValueError, "exactly once"):
                check_source(self.root, split)

    def test_changed_frame_file_invalidates_index(self):
        (self.root / "data").mkdir()
        payload = self.root / "data/frames.parquet"
        payload.write_bytes(b"original")
        index, _ = self.build()
        payload.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "source changed"):
            TemporalDataset(index)

    def test_invalid_selection_or_validation_rejected(self):
        for name, value in (
            ("selection.json", {"usable": False}),
            ("validation.json", {"success": False}),
        ):
            original = (self.root / name).read_text()
            self.save(name, value)
            with self.assertRaisesRegex(ValueError, "validated, usable"):
                check_source(self.root, self.split)
            (self.root / name).write_text(original)
        wrong = copy.deepcopy(self.selection)
        wrong["successful_episodes"].append(1)
        self.save("selection.json", wrong)
        with self.assertRaisesRegex(ValueError, "outcomes"):
            check_source(self.root, self.split)

    def test_terminal_or_failure_reference_tampering_is_rejected(self):
        index, report = self.build()
        for ref in ([0, 6], [1, 2], [2, 2]):
            report["windows"]["train"]["imitation"][0] = ref
            index.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                TemporalDataset(index)[0]

    def test_invalid_window_size_and_no_overwrite(self):
        for history in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "positive integers"):
                prepare(self.root, self.root / "bad.json", self.split, history=history)
        index, _ = self.build()
        with self.assertRaises(FileExistsError):
            prepare(self.root, index, self.split)
