"""Reactive inference boundaries and no-test-access training round trip."""

import json
import unittest
from types import SimpleNamespace

import numpy as np
import test_temporal_data as temporal_fixture

from embodied_agent.reactive import (
    FEATURE_COUNT,
    ReactiveExecutor,
    ReactivePolicy,
    check_experiment,
    observation_features,
    train,
    training_arrays,
)


class ReactiveTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        self.rgb[110:125, 145:160] = [230, 20, 20]
        self.joints = np.array([-1.0, 1.0, 0.0, 0.03, 0.03])

    def test_visual_features_use_current_rgb_and_represent_occlusion(self):
        feature = observation_features(self.rgb, self.joints, True)
        self.assertEqual(feature.shape, (FEATURE_COUNT,))
        self.assertEqual(feature[5], 1)
        np.testing.assert_array_equal(feature[6:11], self.joints.astype(np.float32))
        hidden = observation_features(np.zeros_like(self.rgb), self.joints, False)
        np.testing.assert_array_equal(hidden[:6], np.zeros(6))
        self.assertEqual(hidden[-1], 0)
        moved = np.roll(self.rgb, 20, axis=1)
        self.assertGreater(observation_features(moved, self.joints, True)[0], feature[0])
        with self.assertRaises(ValueError):
            observation_features(self.rgb, np.full(5, np.nan), False)
        with self.assertRaises(ValueError):
            observation_features(self.rgb, self.joints, np.nan)

    def policy(self):
        calls = []

        def predict(history):
            calls.append(history.copy())
            return np.array([[2.0, 2.0, 2.0, 0.0, 0.0]])

        policy = SimpleNamespace(
            metadata={"fps": 25, "history": 4},
            bounds=np.array([[-3.0, 3.0]] * 3 + [[0.0, 0.04]] * 2),
            predict=predict,
        )
        return policy, calls

    def test_real_history_warmup_rate_limits_and_user_stop_latches(self):
        policy, calls = self.policy()
        executor = ReactiveExecutor(policy, self.joints)
        for _ in range(3):
            np.testing.assert_array_equal(executor.step(self.rgb, self.joints, False), self.joints)
        self.assertEqual(len(calls), 0)
        command = executor.step(self.rgb, self.joints, False)
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.all(np.abs(command - self.joints) <= executor.limit + 1e-8))
        command = executor.step(np.roll(self.rgb, 20, axis=1), self.joints, True)
        self.assertEqual(len(calls), 2)
        self.assertGreater(calls[1][-1, 0], calls[0][-1, 0])
        self.assertEqual(calls[1][-1, -1], 1)
        stopped = executor.step(self.rgb, self.joints, False, stop=True)
        np.testing.assert_array_equal(stopped[3:], command[3:])
        np.testing.assert_array_equal(executor.step(self.rgb, self.joints, False), stopped)
        self.assertEqual(len(calls), 2)
        self.assertEqual(executor.reason, "user stop")

    def test_invalid_observation_output_and_timeout_stop(self):
        policy, _ = self.policy()
        executor = ReactiveExecutor(policy, self.joints)
        executor.step(self.rgb, np.full(5, np.nan), False)
        self.assertEqual(executor.state, "stopped")
        policy.predict = lambda history: np.full((1, 5), np.nan)
        executor = ReactiveExecutor(policy, self.joints)
        for _ in range(4):
            executor.step(self.rgb, self.joints, False)
        self.assertEqual(executor.state, "stopped")
        executor = ReactiveExecutor(policy, self.joints, max_seconds=0.04)
        executor.step(self.rgb, self.joints, False)
        executor.step(self.rgb, self.joints, False)
        self.assertEqual(executor.reason, "execution timeout")

    def fixture(self):
        fixture = temporal_fixture.TemporalTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.manifest["model_sha256"] = "fixture-model"
        fixture.save("recording.json", fixture.manifest)
        index, _ = fixture.build()
        config = {
            "development": [e["scenario"] for e in fixture.manifest["episodes"]],
            "split": fixture.split,
            "test": [{"name": "new", "x": 0.39}],
            "seed": 73,
            "epochs": 2,
            "hidden": 16,
            "learning_rate": 0.001,
        }
        config_path = fixture.root / "experiment.json"
        config_path.write_text(json.dumps(config))
        return fixture, index, config_path

    def test_training_excludes_failures_and_test_frames(self):
        fixture, index, config_path = self.fixture()
        fixture.accessed.clear()
        x, y, _joints, _ = training_arrays(index, "train")
        self.assertEqual(x.shape, (4, 3 * FEATURE_COUNT))
        self.assertEqual(y.shape, (4, 2, 5))
        self.assertTrue(all(i < 8 for i in fixture.accessed))  # Failure episode 1 untouched.
        output = fixture.root / "model"
        report = train(index, config_path, output)
        self.assertFalse(report["test_used_for_training"])
        self.assertEqual(report["train_windows"], 4)
        self.assertTrue(all(i < 24 for i in fixture.accessed))
        policy = ReactivePolicy(output)
        actions = policy.predict(x[0].reshape(3, FEATURE_COUNT))
        self.assertEqual(actions.shape, (2, 5))
        self.assertTrue(np.isfinite(actions).all())
        self.assertTrue(np.all(actions >= policy.bounds[:, 0]))
        self.assertTrue(np.all(actions <= policy.bounds[:, 1]))
        np.testing.assert_allclose(policy.mean, x.mean(axis=0))
        with self.assertRaises(FileExistsError):
            train(index, config_path, output)
        with (output / "weights.pt").open("ab") as stream:
            stream.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "integrity"):
            ReactivePolicy(output)

    def test_frozen_experiment_rejects_overlap_and_changed_development(self):
        fixture, _, config_path = self.fixture()
        config = json.loads(config_path.read_text())
        config["test"][0]["x"] = 0.30
        with self.assertRaisesRegex(ValueError, "overlaps"):
            check_experiment(config, fixture.manifest)
        config["development"][0]["x"] = 0.31
        with self.assertRaisesRegex(ValueError, "differ"):
            check_experiment(config, fixture.manifest)
