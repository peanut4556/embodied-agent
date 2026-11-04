import unittest

import numpy as np

from embodied_agent.rigid_reference import RigidReferenceConfirmation

CATALOG = {
    "blue": (0.48, -0.12, 0.004),
    "yellow": (0.64, -0.12, 0.004),
    "magenta": (0.8, 0.12, 0.004),
}
CHECKPOINT = (0.60, 0.12, 0.034)
CENTER = np.mean(list(CATALOG.values()), axis=0)


def rotation(degrees, axis=2):
    angle = np.deg2rad(degrees)
    matrix = np.eye(3)
    a, b = [i for i in range(3) if i != axis]
    matrix[a, a] = matrix[b, b] = np.cos(angle)
    matrix[a, b], matrix[b, a] = -np.sin(angle), np.sin(angle)
    return matrix


def observations(transform):
    return {
        key: transform(np.array(point))
        for key, point in {**CATALOG, "checkpoint": CHECKPOINT}.items()
    }


class RigidReferenceTests(unittest.TestCase):
    def test_corrects_each_rotation_axis_and_translation(self):
        for axis in range(3):
            matrix = rotation(2, axis)
            transform = lambda point, r=matrix: r @ (point - CENTER) + CENTER + (-0.02, 0, 0)
            check = RigidReferenceConfirmation(0, CATALOG, CHECKPOINT)
            target = np.array((0.64, 0, 0.04))
            corrected, detail = check.correct(transform(target), observations(transform))
            np.testing.assert_allclose(corrected, target, atol=1e-12)
            self.assertAlmostEqual(detail["rotation_degrees"], 2)

    def test_checkpoint_detects_out_of_plane_scale_invisible_to_fit(self):
        transform = lambda point: CENTER + (point - CENTER) * (1, 1, 2)
        check = RigidReferenceConfirmation(0, CATALOG, CHECKPOINT)
        corrected, detail = check.correct(
            transform(np.array((0.64, 0, 0.04))), observations(transform)
        )
        self.assertIsNone(corrected)
        self.assertEqual(detail["error"], "independent checkpoint mismatch")

    def test_checkpoint_is_not_used_to_make_fit_look_good(self):
        check = RigidReferenceConfirmation(0, CATALOG, CHECKPOINT)
        refs = observations(lambda point: point)
        refs["checkpoint"][0] += 0.012
        self.assertEqual(
            check.correct((0.64, 0, 0.04), refs)[1]["error"], "independent checkpoint mismatch"
        )

    def test_missing_checkpoint_and_extrapolated_target_reject(self):
        check = RigidReferenceConfirmation(0, CATALOG, CHECKPOINT)
        self.assertFalse(check.correct((0.64, 0, 0.04), CATALOG)[1]["accepted"])
        self.assertEqual(
            check.correct((1, 0, 0.04), observations(lambda p: p))[1]["error"],
            "target outside validated reference neighborhood",
        )

    def test_rotation_and_translation_budgets(self):
        check = RigidReferenceConfirmation(0, CATALOG, CHECKPOINT)
        for matrix, shift in ((rotation(6), np.zeros(3)), (rotation(0), np.array((0.04, 0, 0)))):
            transform = lambda p, r=matrix, t=shift: r @ (p - CENTER) + CENTER + t
            self.assertFalse(
                check.correct(transform(np.array((0.64, 0, 0.04))), observations(transform))[1][
                    "accepted"
                ]
            )

    def test_common_catalog_error_remains_a_counterexample(self):
        shifted = {key: np.array(value) - (0.02, 0, 0) for key, value in CATALOG.items()}
        check = RigidReferenceConfirmation(0, shifted, np.array(CHECKPOINT) - (0.02, 0, 0))
        refs = observations(lambda p: p - (0.02, 0, 0))
        for tick in range(6):
            check.step(tick * 0.2, (0.685, 0, 0.04), (0.685, 0, 0.04), refs, refs)
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_bad_checkpoint_cannot_preserve_earlier_stability(self):
        check = RigidReferenceConfirmation(0, CATALOG, CHECKPOINT)
        for tick in range(16):
            refs = observations(lambda p: p)
            if tick in (4, 8, 12):
                refs.pop("checkpoint")
            check.step(tick * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04), refs, refs)
        self.assertEqual(check.status, "unconfirmed")

    def test_checkpoint_must_be_independent_and_off_plane(self):
        with self.assertRaises(ValueError):
            RigidReferenceConfirmation(0, CATALOG, (0.6, 0.1, 0.004))
        with self.assertRaises(ValueError):
            RigidReferenceConfirmation(
                0,
                {"checkpoint": CATALOG["blue"], "b": CATALOG["yellow"], "c": CATALOG["magenta"]},
                CHECKPOINT,
            )
