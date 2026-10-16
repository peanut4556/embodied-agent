import copy
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from embodied_agent.physics import PhysicsWorld
from scripts.evaluate_policy_generalization import apply_visual, cases, paired_checks, summarize


class PolicyGeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("config/policy-generalization.json").read_text())

    def test_frozen_grid_excludes_known_positions(self):
        grid = cases(self.config)
        self.assertEqual(len(grid), 24)
        self.assertEqual(len({c["case_id"] for c in grid}), 24)
        self.assertFalse(set(self.config["positions"]) & set(self.config["excluded_positions"]))
        for x in self.config["positions"]:
            self.assertEqual(len([c for c in grid if c["x"] == x]), 4)

    def test_reject_overlap_duplicates_nonfinite_and_invalid_design(self):
        for key, value in [
            ("positions", [0.302]),
            ("positions", [0.283, 0.283]),
            ("positions", [float("nan")]),
            ("positions", [0.8]),
            ("modes", ["guarded"]),
            ("max_seconds", 60),
        ]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                cases(config)

    def test_reject_invalid_visuals(self):
        for key, value in [("camera_dx", 0.1), ("light_scale", 0), ("camera_dx", float("nan"))]:
            config = copy.deepcopy(self.config)
            config["visuals"][1][key] = value
            with self.assertRaises(ValueError):
                cases(config)
        self.config["visuals"][1]["name"] = "nominal"
        with self.assertRaises(ValueError):
            cases(self.config)

    def test_render_perturbation_preserves_physics(self):
        a, b = PhysicsWorld(), PhysicsWorld()
        try:
            diffuse = b.model.light_diffuse.copy()
            cam = b.model.camera("perception").id
            camera = b.model.cam_pos[cam].copy()
            apply_visual(b, {"light_scale": 0.55, "camera_dx": 0.015})
            np.testing.assert_allclose(b.model.light_diffuse, diffuse * 0.55)
            np.testing.assert_allclose(b.model.cam_pos[cam], camera + [0.015, 0, 0])
            for _ in range(200):
                mujoco.mj_step(a.model, a.data)
                mujoco.mj_step(b.model, b.data)
            np.testing.assert_array_equal(a.data.qpos, b.data.qpos)
            np.testing.assert_array_equal(a.data.qvel, b.data.qvel)
        finally:
            a.close()
            b.close()

    def test_pairing_detects_different_prefix_and_initial_state(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            actions = np.zeros((4, 5))
            np.savez(output / "a.npz", actions=actions, qpos=actions, features=actions)
            changed = actions.copy()
            changed[1, 0] = 1
            np.savez(output / "b.npz", actions=changed, qpos=actions, features=actions)
            base = {
                "case_id": "fixture",
                "mode": "policy",
                "initial_state": {"x": 0.3},
                "initial_image_sha256": "same",
                "trace": "a.npz",
                "quality": {"verified_pick_place": True},
            }
            guarded = {
                **base,
                "mode": "guarded",
                "trace": "b.npz",
                "takeover_tick": 2,
                "initial_state": {"x": 0.4},
                "quality": {"verified_pick_place": False},
            }
            result = paired_checks([base, guarded], output)[0]
            self.assertFalse(result["equal_pre_takeover_prefix"])
            self.assertFalse(result["equal_initial_state"])
            self.assertTrue(result["baseline_success_lost"])
            guarded["takeover_tick"] = 1
            self.assertTrue(paired_checks([base, guarded], output)[0]["equal_pre_takeover_prefix"])

    def test_summary_separates_policy_and_expert_success(self):
        rows = [
            {
                "mode": "policy",
                "visual": "dim",
                "takeover_tick": None,
                "failure": "no_stable_grasp",
                "quality": {"verified_pick_place": False},
            },
            {
                "mode": "guarded",
                "visual": "dim",
                "takeover_tick": 79,
                "failure": "none",
                "quality": {"verified_pick_place": True},
            },
        ]
        summary = summarize(rows)
        self.assertEqual(summary["policy"]["all"]["successes"], 0)
        self.assertEqual(summary["guarded"]["all"]["successful_after_takeover"], 1)
        self.assertEqual(summary["policy"]["dim"]["cases"], 1)
        self.assertEqual(summary["guarded"]["dim"]["success_rate"], 1)


if __name__ == "__main__":
    unittest.main()
