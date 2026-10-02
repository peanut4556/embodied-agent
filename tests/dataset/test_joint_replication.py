"""Replications reject changed paired budgets and identify duplicate controls."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.evaluate_joint_training import evaluate
from scripts.replicate_joint_training import same_parameters


class JointReplicationTests(unittest.TestCase):
    def test_parameter_identity_uses_tensor_values(self):
        first = SimpleNamespace(model=torch.nn.Linear(2, 2))
        second = SimpleNamespace(model=copy.deepcopy(first.model))
        self.assertTrue(same_parameters(first, second))
        with torch.no_grad():
            second.model.weight[0, 0] += 0.01
        self.assertFalse(same_parameters(first, second))

    def test_mismatched_seeds_rejected_before_physics(self):
        configs = [
            {"seed": 101, "joint_augmentation": None},
            {"seed": 211, "joint_augmentation": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "evaluation"
            with (
                patch(
                    "scripts.evaluate_joint_training.verify_run",
                    side_effect=[({}, c) for c in configs],
                ),
                self.assertRaisesRegex(ValueError, "beyond augmentation"),
            ):
                evaluate("control", "augmented", target)
            self.assertFalse(target.exists())

    def test_mismatched_data_rejected_before_physics(self):
        configs = [
            {"seed": 101, "joint_augmentation": None},
            {"seed": 101, "joint_augmentation": {}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "evaluation"
            runs = [({"source_sha256": str(i)}, c) for i, c in enumerate(configs)]
            with (
                patch("scripts.evaluate_joint_training.verify_run", side_effect=runs),
                self.assertRaisesRegex(ValueError, "inputs or parent"),
            ):
                evaluate("control", "augmented", target)
            self.assertFalse(target.exists())
