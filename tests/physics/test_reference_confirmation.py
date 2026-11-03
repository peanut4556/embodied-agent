import unittest

from embodied_agent.reference_confirmation import ReferenceConfirmation

CATALOG = {
    "blue": (0.5, -0.17, 0.004),
    "yellow": (0.64, -0.19, 0.004),
    "magenta": (0.78, -0.15, 0.004),
}


def shifted(dx):
    return {key: (point[0] + dx, point[1], point[2]) for key, point in CATALOG.items()}


class ReferenceConfirmationTests(unittest.TestCase):
    def test_common_bias_corrects_inside_and_rejects_outside_goal(self):
        for truth, expected in (
            (0.64, "confirmed_in_tray"),
            (0.705, "outside_confirmation_region"),
        ):
            check = ReferenceConfirmation(0, CATALOG)
            for tick in range(16):
                check.step(
                    tick * 0.2,
                    (truth - 0.02, 0, 0.04),
                    (truth - 0.02, 0, 0.04),
                    shifted(-0.02),
                    shifted(-0.02),
                )
            self.assertEqual(check.status, expected)

    def test_missing_inconsistent_or_excessive_references_reject(self):
        inconsistent = shifted(0)
        inconsistent["blue"] = (0.512, -0.17, 0.004)
        for references in (None, {}, inconsistent, shifted(-0.06), {**CATALOG, "extra": (0, 0, 0)}):
            check = ReferenceConfirmation(0, CATALOG)
            for tick in range(16):
                check.step(tick * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04), references, CATALOG)
            self.assertEqual(check.status, "unconfirmed")
            self.assertFalse(check.rows[0]["primary"]["accepted"])

    def test_missing_sample_breaks_stability_without_new_deadline(self):
        check = ReferenceConfirmation(0, CATALOG)
        for tick in range(16):
            refs = None if tick in (4, 8, 12) else CATALOG
            check.step(tick * 0.2, (0.64, 0, 0.04), (0.64, 0, 0.04), refs, CATALOG)
        self.assertEqual(check.status, "unconfirmed")

    def test_reference_catalog_common_error_remains_undetectable(self):
        check = ReferenceConfirmation(0, shifted(-0.02))
        for tick in range(6):
            check.step(
                tick * 0.2, (0.685, 0, 0.04), (0.685, 0, 0.04), shifted(-0.02), shifted(-0.02)
            )
        self.assertEqual(check.status, "confirmed_in_tray")

    def test_no_target_fitting_and_snapshot_isolation(self):
        check = ReferenceConfirmation(0, CATALOG)
        first, detail = check.correct((0.5, 0, 0.04), shifted(-0.02))
        second, other = check.correct((0.8, 0, 0.04), shifted(-0.02))
        self.assertEqual(detail, other)
        self.assertAlmostEqual(second[0] - first[0], 0.3)
        snapshot = check.result()
        snapshot["reference_points"].clear()
        self.assertEqual(len(check.reference_points), 3)

    def test_invalid_catalog_and_bad_coordinates(self):
        for catalog in ({}, {"a": (0, 0, 0), "b": (1, 0, 0), "c": (2, 0, 0)}):
            with self.assertRaises(ValueError):
                ReferenceConfirmation(0, catalog)
        check = ReferenceConfirmation(0, CATALOG)
        for point in (None, (float("nan"), 0, 0), (10**400, 0, 0)):
            self.assertIsNone(check.correct(point, CATALOG)[0])

    def test_stop_and_invalid_time_do_not_record_evidence(self):
        check = ReferenceConfirmation(0, CATALOG)
        with self.assertRaises(ValueError):
            check.step(float("nan"), (0.64, 0, 0.04), (0.64, 0, 0.04), CATALOG, CATALOG)
        self.assertEqual(check.rows, [])
        check.step(None, None, None, None, None, stop=True)
        self.assertEqual(check.status, "cancelled")
