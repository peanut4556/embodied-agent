"""Reject validation/duplicate/unmasked targets and retain original supervision."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from embodied_agent.local_targets import auxiliary_batch


class LocalTargetTests(unittest.TestCase):
    def setUp(self):
        self.policy = SimpleNamespace(
            metadata={"weights_sha256": "teacher"}, bounds=np.array([[-1.0, 1.0]] * 5)
        )
        self.episodes = [
            {
                "episode": 0,
                "is_correction": True,
                "x": np.zeros((2, 12)),
                "loss_mask": np.array([False, True]),
            }
        ]
        self.y = torch.zeros((1, 2, 5))
        self.row = {
            "group": "train",
            "episode": 0,
            "tick": 1,
            "projected_command": [0.2] * 5,
            "loss_before": 1.0,
            "loss_after": 0.5,
            "target_replay_max_error": 0.0,
        }
        self.report = {
            "status": "complete",
            "weights_sha256": "teacher",
            "source_sha256": "data",
            "rows": [self.row],
        }
        self.config = {"path": "unused", "sha256": "hash", "weight": 0.1}

    def load(self, report):
        with (
            patch("embodied_agent.local_targets.digest", return_value="hash"),
            patch("embodied_agent.local_targets.payload_digest", return_value="data"),
            patch("embodied_agent.local_targets.read_json", return_value=report),
        ):
            return auxiliary_batch(self.episodes, self.y, self.policy, self.config, ["root"])

    def test_separate_targets_leave_originals_and_prefix_unchanged(self):
        target, mask, meta = self.load(self.report)
        torch.testing.assert_close(target[0, 1], torch.full((5,), 0.6))
        self.assertEqual(mask.sum(), 1)
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(self.y.sum(), 0)
        self.assertEqual(meta["accepted"], 1)

    def test_validation_duplicate_and_unsupervised_rows_rejected(self):
        for change in ({"group": "validation"}, {"tick": 0}, {"episode": 9}):
            report = copy.deepcopy(self.report)
            report["rows"][0].update(change)
            with self.assertRaises(ValueError):
                self.load(report)
        report = copy.deepcopy(self.report)
        report["rows"].append(copy.deepcopy(report["rows"][0]))
        with self.assertRaises(ValueError):
            self.load(report)

    def test_nonimproving_target_rejected_and_teacher_lineage_checked(self):
        report = copy.deepcopy(self.report)
        report["rows"][0]["loss_after"] = 2.0
        with self.assertRaisesRegex(ValueError, "no accepted"):
            self.load(report)
        report = copy.deepcopy(self.report)
        report["weights_sha256"] = "another"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.load(report)

    def test_nonfinite_replay_evidence_rejected(self):
        report = copy.deepcopy(self.report)
        report["rows"][0]["target_replay_max_error"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid local target"):
            self.load(report)
