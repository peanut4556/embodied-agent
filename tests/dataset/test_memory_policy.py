"""Causal memory, episode isolation and physical-selection gates."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_temporal_data as temporal_fixture
import torch

from embodied_agent.memory_evaluation import select_candidate, selected_policy
from embodied_agent.memory_policy import (
    MemoryPolicy,
    MemorySession,
    check_config,
    make_network,
    masked_loss,
    padded_batch,
    sequences,
    train,
)
from embodied_agent.reactive import FEATURE_COUNT, ReactiveExecutor


class MemoryTests(unittest.TestCase):
    def test_network_prefix_does_not_read_future_and_stepwise_matches_batch(self):
        torch.manual_seed(3)
        model = make_network(16).eval()
        x = torch.randn(1, 12, FEATURE_COUNT)
        with torch.no_grad():
            whole = model(x)[0]
            hidden, outputs = None, []
            for t in range(12):
                out, hidden = model(x[:, t : t + 1], hidden)
                outputs.append(out)
            changed = x.clone()
            changed[:, 6:] = 100
            modified = model(changed)[0]
        torch.testing.assert_close(torch.cat(outputs, dim=1), whole, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(modified[:, :6], whole[:, :6])

    def policy(self):
        torch.manual_seed(3)
        return SimpleNamespace(
            metadata={"history": 1, "fps": 25},
            bounds=np.array([[-3.0, 3.0]] * 3 + [[0.0, 0.04]] * 2),
            mean=np.zeros(FEATURE_COUNT),
            scale=np.ones(FEATURE_COUNT),
            model=make_network(16).eval(),
        )

    def test_sessions_reset_and_cannot_contaminate_another_execution(self):
        policy = self.policy()
        first, second, reference = (MemorySession(policy) for _ in range(3))
        x = np.full((1, FEATURE_COUNT), 0.1, dtype=np.float32)
        first.predict(x)
        first_hidden = first.hidden.clone()
        first.predict(x * 2)
        self.assertFalse(torch.allclose(first.hidden, first_hidden))
        np.testing.assert_allclose(second.predict(x), reference.predict(x))
        torch.testing.assert_close(second.hidden, first_hidden)
        with self.assertRaises(ValueError):
            first.predict(np.full_like(x, np.nan))
        # Stop must stop advancing memory and preserve the command.
        executor = ReactiveExecutor(second, np.zeros(5))
        held = executor.step(np.zeros((240, 320, 3), dtype=np.uint8), np.zeros(5), False, stop=True)
        before = second.hidden.clone()
        np.testing.assert_array_equal(executor.step(None, None, None), held)
        torch.testing.assert_close(second.hidden, before)

    def test_padding_is_masked_out_of_loss(self):
        episodes = [{"x": np.zeros((n, FEATURE_COUNT)), "y": np.zeros((n, 5))} for n in (2, 4)]
        _, y, mask = padded_batch(
            episodes, np.zeros(FEATURE_COUNT), np.ones(FEATURE_COUNT), np.array([[-1.0, 1.0]] * 5)
        )
        pred = torch.zeros_like(y)
        original = masked_loss(pred, y, mask)
        y[0, 2:] = 1000
        torch.testing.assert_close(masked_loss(pred, y, mask), original)
        self.assertEqual(mask.sum().item(), 6)

    def fixture(self):
        fixture = temporal_fixture.TemporalTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.manifest["episodes"] = fixture.manifest["episodes"][:3]
        fixture.manifest["model_sha256"] = "test-physics"
        fixture.selection.update(successful_episodes=[0, 2], independent_test_episodes=[])
        fixture.save("recording.json", fixture.manifest)
        fixture.save("selection.json", fixture.selection)
        # Retain the fixture's hard error on any read at index >=24.
        original = fixture.frame

        def frame(index):
            row = original(index)
            row["observation.state"] = torch.tensor([-1.0, 1.0, 0.0, 0.03, 0.03])
            row["action"] = torch.tensor([-1.0 + (index % 8) * 0.001, 1.0, 0.0, 0.03, 0.03])
            return row

        # The constructor is already patched by TemporalTests.setUp.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset.return_value.__len__.return_value = 24
        LeRobotDataset.return_value.__getitem__.side_effect = frame
        config = {
            "split": {"train": [0, 1], "validation": [2], "test": []},
            "candidate_epochs": [1, 2],
            "hidden": 16,
            "learning_rate": 0.001,
            "max_seconds": 24,
            "previous_test_positions": [0.36],
            "seed": 73,
            "validation_cases": [{"name": "dev", "x": 0.34}],
            "test": [{"name": "new", "x": 0.39}],
            "selection": "success then mse then epoch",
        }
        path = fixture.root / "experiment.json"
        path.write_text(json.dumps(config))
        return fixture, config, path

    def test_successful_sequences_only_and_training_roundtrip(self):
        fixture, config, path = self.fixture()
        seq = sequences(fixture.root, config["split"], "train")
        self.assertEqual([e["episode"] for e in seq], [0])
        self.assertTrue(all(i < 8 for i in fixture.accessed))
        self.assertEqual(len(seq[0]["x"]), 7)  # Completed frame excluded.
        with self.assertRaises(ValueError):
            sequences(fixture.root, config["split"], "test")
        run_path = fixture.root / "run"
        report = train(fixture.root, path, run_path)
        self.assertEqual(report["status"], "trained")
        self.assertFalse(report["test_used_for_training"])
        self.assertTrue(all(i < 8 or 16 <= i < 24 for i in fixture.accessed))
        policy = MemoryPolicy(run_path / "epoch-2")
        np.testing.assert_allclose(policy.mean, seq[0]["x"].mean(0))
        output = policy.session().predict(seq[0]["x"][:1])
        self.assertEqual(output.shape, (1, 5))
        self.assertTrue(np.isfinite(output).all())
        with self.assertRaises(FileExistsError):
            train(fixture.root, path, run_path)
        with (run_path / "epoch-2/weights.pt").open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "integrity"):
            MemoryPolicy(run_path / "epoch-2")

    def test_physical_success_outranks_lower_offline_error(self):
        a = {"epoch": 1, "task_success": 0, "validation_mse": 0.00001}
        b = {"epoch": 2, "task_success": 2, "validation_mse": 0.1}
        self.assertEqual(select_candidate([a, b]), b)
        self.assertEqual(select_candidate([b, {**b, "epoch": 3}]), b)
        with self.assertRaises(ValueError):
            select_candidate([])

    def test_config_rejects_test_reuse_and_training_scenes_for_selection(self):
        fixture, config, _ = self.fixture()
        check_config(config, fixture.manifest)
        config["test"][0]["x"] = 0.36
        with self.assertRaisesRegex(ValueError, "overlaps"):
            check_config(config, fixture.manifest)
        config["test"][0]["x"] = 0.39
        config["validation_cases"][0]["x"] = 0.30
        with self.assertRaisesRegex(ValueError, "validation"):
            check_config(config, fixture.manifest)

    def test_test_entry_rejects_incomplete_selection_before_loading_policy(self):
        fixture, config, _ = self.fixture()
        selection = fixture.root / "selection-attempt.json"
        selection.write_text(json.dumps({"status": "running"}))
        with (
            patch("embodied_agent.memory_evaluation.verify_run", return_value=({}, config)),
            self.assertRaisesRegex(ValueError, "complete development"),
        ):
            selected_policy(fixture.root, selection)

    def test_zero_success_candidate_does_not_open_reserved_test(self):
        fixture, config, _ = self.fixture()
        candidate = {
            "directory": "epoch-1",
            "epoch": 1,
            "validation_mse": 0.01,
            "weights_sha256": "weights",
        }
        run = {
            "candidates": [candidate],
            "experiment_sha256": "experiment",
            "source_sha256": "data",
        }
        selection = {
            "status": "selected",
            "test_cases_executed": False,
            "experiment_sha256": "experiment",
            "source_sha256": "data",
            "candidates": [
                {**candidate, "task_success": 0, "results": [{"case": "dev", "success": False}]}
            ],
            "selected": "epoch-1",
            "selected_weights_sha256": "weights",
        }
        path = fixture.root / "zero-success.json"
        path.write_text(json.dumps(selection))
        with (
            patch("embodied_agent.memory_evaluation.verify_run", return_value=(run, config)),
            self.assertRaisesRegex(ValueError, "at least one development success"),
        ):
            selected_policy(fixture.root, path)
