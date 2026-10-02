"""Augmentation is reproducible, training-only and limited to real joint frames."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from embodied_agent.correction_finetune import augment_joints


class JointAugmentationTests(unittest.TestCase):
    def test_only_real_joint_features_change_reproducibly(self):
        policy = SimpleNamespace(
            mean=np.zeros(12), scale=np.ones(12), bounds=np.array([[-1.0, 1.0]] * 5)
        )
        x = torch.zeros((2, 5, 12))
        x[:, :, 11] = 1
        config = {"probability": 1.0, "arm_radians": 0.01, "finger_meters": 0.001}
        first = augment_joints(x, [2, 5], policy, config, torch.Generator().manual_seed(3))
        second = augment_joints(x, [2, 5], policy, config, torch.Generator().manual_seed(3))
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(first[:, :, :6], x[:, :, :6])
        torch.testing.assert_close(first[:, :, 11], x[:, :, 11])
        torch.testing.assert_close(first[0, 2:], x[0, 2:])
        self.assertTrue((first[:, :, :11] != x[:, :, :11]).any())
        self.assertLessEqual(float(first[:, :, 6:9].abs().max()), 0.01)
        self.assertLessEqual(float(first[:, :, 9:11].abs().max()), 0.001)
        self.assertEqual(float(x[:, :, :11].sum()), 0.0)

    def test_disabled_is_identity_and_physical_bounds_apply_with_normalization(self):
        policy = SimpleNamespace(
            mean=np.ones(12), scale=np.full(12, 0.5), bounds=np.array([[0.0, 1.0]] * 5)
        )
        x = torch.zeros((1, 4, 12))
        self.assertIs(augment_joints(x, [4], policy, None, torch.Generator()), x)
        cfg = {"probability": 0.0, "arm_radians": 0.01, "finger_meters": 0.001}
        torch.testing.assert_close(augment_joints(x, [4], policy, cfg, torch.Generator()), x)
        cfg["probability"] = 1.0
        y = augment_joints(x, [4], policy, cfg, torch.Generator())
        physical = y[:, :, 6:11] * 0.5 + 1
        self.assertTrue(torch.all(physical <= 1))
        self.assertTrue(torch.all(physical >= 0))
        cfg["arm_radians"] = -0.1
        with self.assertRaises(ValueError):
            augment_joints(x, [4], policy, cfg, torch.Generator())
