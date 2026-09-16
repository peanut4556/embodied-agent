"""Masked fine-tuning and stricter physical selection regression tests."""

import copy
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from embodied_agent.correction_finetune import batch, check_groups
from embodied_agent.memory_evaluation import select_candidate
from embodied_agent.memory_policy import masked_loss


class CorrectionFinetuneTests(unittest.TestCase):
    def test_rollin_and_settle_targets_have_zero_gradient(self):
        policy = SimpleNamespace(
            mean=np.zeros(12), scale=np.ones(12), bounds=np.array([[-1.0, 1.0]] * 5)
        )
        ep = {
            "x": np.ones((5, 12)),
            "y": np.zeros((5, 5)),
            "loss_mask": np.array([False, False, True, True, False]),
        }
        x, y, mask = batch([ep], policy)
        self.assertTrue(torch.all(x == 1))
        pred = torch.ones_like(y, requires_grad=True)
        loss = masked_loss(pred, y, mask)
        loss.backward()
        self.assertTrue(torch.all(pred.grad[0, [0, 1, 4]] == 0))
        self.assertTrue(torch.all(pred.grad[0, [2, 3]] != 0))
        y[0, [0, 1, 4]] = 999
        self.assertEqual(float(masked_loss(pred, y, mask).detach()), float(loss.detach()))
        ep["loss_mask"][:] = False
        with self.assertRaises(ValueError):
            batch([ep], policy)

    def test_cross_dataset_split_and_test_leakage(self):
        split = {"train": [0], "validation": [1], "test": []}
        base = {
            "fps": 25,
            "model_sha256": "scene",
            "episodes": [{"scenario": {"x": x}} for x in [0.30, 0.34]],
        }
        corrections = dict(**base, split=split)
        policy = SimpleNamespace(metadata={"fps": 25, "model_sha256": "scene"})
        config = {"previous_test_positions": [0.36], "test": [{"x": 0.39}]}
        check_groups(base, split, [corrections], config, policy)
        crossed = copy.deepcopy(corrections)
        crossed["episodes"][0]["scenario"]["x"] = 0.34
        with self.assertRaisesRegex(ValueError, "overlap"):
            check_groups(base, split, [crossed], config, policy)
        config["test"][0]["x"] = 0.36
        with self.assertRaisesRegex(ValueError, "overlaps"):
            check_groups(base, split, [corrections], config, policy)

    def test_verified_grasp_outranks_endpoint_only_success(self):
        pushed = {
            "epoch": 100,
            "task_success": 3,
            "verified_pick_place": 0,
            "validation_mse": 0.001,
        }
        grasped = {"epoch": 300, "task_success": 1, "verified_pick_place": 1, "validation_mse": 0.1}
        self.assertEqual(select_candidate([pushed, grasped], "verified_pick_place"), grasped)
        self.assertEqual(select_candidate([pushed, grasped]), pushed)

    def test_endpoint_success_cannot_open_qualified_test_gate(self):
        from unittest.mock import patch

        from embodied_agent.memory_evaluation import selected_policy

        candidate = {
            "directory": "epoch-100",
            "epoch": 100,
            "validation_mse": 0.1,
            "weights_sha256": "w",
        }
        run = {"candidates": [candidate], "experiment_sha256": "e", "source_sha256": "s"}
        config = {"selection_metric": "verified_pick_place", "validation_cases": [{"name": "dev"}]}
        selection = {
            "status": "selected",
            "test_cases_executed": False,
            "experiment_sha256": "e",
            "source_sha256": "s",
            "selected": "epoch-100",
            "selected_weights_sha256": "w",
            "candidates": [
                {
                    **candidate,
                    "task_success": 1,
                    "verified_pick_place": 0,
                    "results": [
                        {"case": "dev", "success": True, "quality": {"verified_pick_place": False}}
                    ],
                }
            ],
        }
        with (
            patch("embodied_agent.memory_evaluation.verify_run", return_value=(run, config)),
            patch("embodied_agent.memory_evaluation.read_json", return_value=selection),
            self.assertRaisesRegex(ValueError, "at least one development success"),
        ):
            selected_policy("unused", "unused")
