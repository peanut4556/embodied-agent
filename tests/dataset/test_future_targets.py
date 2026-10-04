"""Completion requires a verified grasp chain; masks affect future labels only."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from embodied_agent.future_targets import completion_flags, future_batch


class FutureTargetTests(unittest.TestCase):
    def test_inside_box_alone_is_not_qualified_completion(self):
        row = {
            "holding": False,
            "block_xyz": [0.64, 0.0, 0.04],
            "command": [0.0, 0.0, 0.0, 0.04, 0.04],
            "inside_box": True,
        }
        self.assertFalse(any(completion_flags([row] * 30)))

    def test_completion_after_grasp_lift_release_and_stability_only(self):
        holding = {
            "holding": True,
            "block_xyz": [0.4, 0.0, 0.2],
            "command": [0.0] * 5,
            "inside_box": False,
        }
        opening = {**holding, "block_xyz": [0.64, 0.0, 0.2], "command": [0.0, 0.0, 0.0, 0.04, 0.04]}
        goal = {**opening, "holding": False, "block_xyz": [0.64, 0.0, 0.04], "inside_box": True}
        rows = [holding] * 3 + [opening] + [goal] * 25
        flags = completion_flags(rows)
        self.assertFalse(any(flags[:-1]))
        self.assertTrue(flags[-1])
        self.assertFalse(completion_flags(rows + [{**goal, "inside_box": False}])[-1])

    def fixture(self):
        row = {
            "episode": 0,
            "group": "train",
            "takeover_tick": 1,
            "horizon_ticks": 1,
            "expert_object_target_m": [0.64, 0.0, 0.04],
            "expert_holding_target": False,
            "expert_inside_box": True,
            "autonomous_verified_complete": False,
        }
        pack = {
            "group": "train",
            "source_sha256": "data",
            "weights_sha256": "model",
            "horizons": [1, 25, 100, 200, 300],
            "targets": [row, {**row, "horizon_ticks": 25, "autonomous_verified_complete": True}],
        }
        policy = SimpleNamespace(metadata={"weights_sha256": "model"})
        episodes = [
            {
                "episode": 0,
                "is_correction": True,
                "x": np.zeros((2, 12)),
                "loss_mask": np.array([False, True]),
            }
        ]
        return pack, policy, episodes

    def load(self, pack, masked):
        _, policy, episodes = self.fixture()
        with (
            patch("embodied_agent.future_targets.digest", return_value="hash"),
            patch("embodied_agent.future_targets.payload_digest", return_value="data"),
            patch("embodied_agent.future_targets.read_json", return_value=pack),
        ):
            return future_batch(
                episodes,
                torch.zeros((1, 2, 12)),
                policy,
                {"path": "unused", "sha256": "hash", "weight": 0.1, "completion_mask": masked},
                ["source"],
            )

    def test_masked_completed_target_has_zero_gradient(self):
        pack, _, _ = self.fixture()
        target, mask, report = self.load(pack, True)
        self.assertEqual(report["protected"], 1)
        self.assertEqual(mask.sum(), 5)
        prediction = torch.zeros_like(target, requires_grad=True)
        (((prediction - target) ** 2 * mask).sum() / mask.sum()).backward()
        self.assertTrue(torch.all(prediction.grad[0, 1, 5:10] == 0))
        self.assertTrue(torch.any(prediction.grad[0, 1, :5] != 0))
        self.assertEqual(self.load(pack, False)[1].sum(), 10)

    def test_validation_labels_rejected(self):
        pack, _, _ = self.fixture()
        pack = copy.deepcopy(pack)
        pack["targets"][0]["group"] = "validation"
        with self.assertRaises(ValueError):
            self.load(pack, True)
