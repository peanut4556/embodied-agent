"""Reject replay leakage, corruption and a different parent's targets."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from embodied_agent.success_replay import replay_batch


class SuccessReplayTests(unittest.TestCase):
    def fixture(self):
        policy = SimpleNamespace(
            metadata={
                "weights_sha256": "parent",
                "normalization_sha256": "norm",
                "model_sha256": "physics",
                "fps": 25,
            },
            mean=np.zeros(12),
            scale=np.ones(12),
            model=lambda x: (torch.ones((*x.shape[:2], 5)), None),
        )
        pack = {
            **policy.metadata,
            "group": "train",
            "position": 0.302,
            "result": {"quality": {"verified_pick_place": True}},
            "observations": np.zeros((3, 12)).tolist(),
            "normalized_targets": np.ones((3, 5)).tolist(),
        }
        return policy, pack

    def load(self, pack, validation=(), digest_value="hash"):
        policy, _ = self.fixture()
        with (
            patch("embodied_agent.success_replay.digest", return_value=digest_value),
            patch("embodied_agent.success_replay.read_json", return_value=pack),
        ):
            return replay_batch(
                {"path": "unused", "sha256": "hash", "weight": 1.0},
                policy,
                {0.302},
                set(validation),
            )

    def test_causal_full_sequence_and_teacher_targets_retained(self):
        _, pack = self.fixture()
        x, y, report = self.load(pack)
        self.assertEqual(x.shape, (1, 3, 12))
        self.assertEqual(report["frames"], 3)
        prediction = y.clone().requires_grad_()
        ((prediction - y).square().mean()).backward()
        self.assertTrue(torch.all(prediction.grad == 0))
        self.assertFalse(y.requires_grad)

    def test_validation_and_unqualified_rollouts_rejected(self):
        _, pack = self.fixture()
        with self.assertRaises(ValueError):
            self.load(pack, validation=(0.302,))
        for group, success in [("validation", True), ("train", False)]:
            bad = copy.deepcopy(pack)
            bad["group"] = group
            bad["result"]["quality"]["verified_pick_place"] = success
            with self.assertRaises(ValueError):
                self.load(bad)

    def test_tampering_and_wrong_parent_rejected(self):
        _, pack = self.fixture()
        with self.assertRaises(ValueError):
            self.load(pack, digest_value="changed")
        pack["weights_sha256"] = "other"
        with self.assertRaises(ValueError):
            self.load(pack)

    def test_nonfinite_and_wrong_teacher_targets_rejected(self):
        _, pack = self.fixture()
        for value in (float("nan"), 0.5):
            pack["normalized_targets"][0][0] = value
            with self.assertRaises(ValueError):
                self.load(pack)
