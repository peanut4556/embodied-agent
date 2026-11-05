import copy
import unittest

from embodied_agent.profile_admission import QualifiedViewConfirmation, admit_profiles


class ProfileAdmissionTests(unittest.TestCase):
    def rows(self, count=8):
        return [
            {
                "id": str(i),
                "samples": [
                    {
                        "truth_xyz": [0.64, 0, 0.045],
                        "profiles": {
                            "coarse": {"primary": [0.643, 0, 0.045], "secondary": [0.64, 0, 0.045]},
                            "detail": {
                                "primary": [0.6405, 0, 0.045],
                                "secondary": [0.64, 0, 0.045],
                            },
                        },
                    }
                ],
            }
            for i in range(count)
        ]

    def admit(self, rows=None):
        return admit_profiles(
            {"coarse": (0.002,) * 3, "detail": (0.001,) * 3},
            ["calibration"],
            self.rows() if rows is None else rows,
        )

    def test_violation_excludes_only_affected_profile(self):
        result = self.admit()
        self.assertEqual(set(dict(result.profiles)), {"detail"})
        self.assertEqual(result.result()["audits"]["coarse"]["exceeded_views"], 8)
        self.assertFalse(result.result()["certified"])

    def test_repeated_frames_cannot_inflate_case_count(self):
        rows = self.rows(1)
        rows[0]["samples"] *= 100
        self.assertEqual(self.admit(rows).profiles, ())

    def test_duplicate_and_calibration_overlap_rejected(self):
        for identity in ("calibration", "1"):
            rows = self.rows()
            rows[0]["id"] = identity
            with self.assertRaises(ValueError):
                self.admit(rows)

    def test_valid_half_of_incomplete_pair_is_still_audited(self):
        rows = self.rows(9)
        rows[0]["samples"][0]["profiles"]["detail"] = {
            "primary": [0.65, 0, 0.045],
            "secondary": None,
        }
        result = self.admit(rows)
        self.assertEqual(result.profiles, ())
        self.assertEqual(result.result()["audits"]["detail"]["exceeded_views"], 1)

    def test_empty_or_incomplete_evidence_cannot_admit(self):
        self.assertEqual(self.admit([]).profiles, ())
        rows = self.rows()
        rows[0]["samples"] = []
        with self.assertRaises(ValueError):
            self.admit(rows)

    def test_nonfinite_or_unrepresented_profile_rejected(self):
        for bad in ("nonfinite", "missing"):
            rows = self.rows()
            profiles = rows[0]["samples"][0]["profiles"]
            if bad == "nonfinite":
                profiles["detail"]["primary"][0] = float("nan")
            else:
                profiles.pop("detail")
            with self.assertRaises(ValueError):
                self.admit(rows)

    def test_runtime_never_requests_rejected_initial_profile(self):
        check = QualifiedViewConfirmation(0, self.admit())
        calls = []

        def locate(profile):
            calls.append(profile)
            return (0.64, 0, 0.045), (0.64, 0, 0.045)

        for tick in range(6):
            check.step(tick * 0.2, locate)
        self.assertEqual(check.status, "confirmed_in_tray")
        self.assertEqual(set(calls), {"detail"})

    def test_no_eligible_profiles_never_acquire(self):
        check = QualifiedViewConfirmation(0, self.admit([]))

        def forbidden(profile):
            self.fail("no acquisition allowed")

        self.assertEqual(check.step(0, forbidden), "unconfirmed")
        self.assertEqual(check.result()["reason"], "no_qualified_profiles")

    def test_future_evidence_can_revoke_previously_eligible_profile(self):
        rows = self.rows()
        original = self.admit(rows)
        new = self.rows(1)[0]
        new["id"] = "future"
        new["samples"][0]["profiles"]["detail"]["primary"] = [0.65, 0, 0.045]
        revoked = self.admit(rows + [new])
        self.assertEqual(revoked.profiles, ())
        self.assertEqual(set(dict(original.profiles)), {"detail"})

    def test_snapshot_and_source_changes_do_not_mutate_admission(self):
        rows = self.rows()
        result = self.admit(rows)
        baseline = copy.deepcopy(result.result())
        rows[0]["samples"].clear()
        snapshot = result.result()
        snapshot["audits"].clear()
        self.assertEqual(result.result(), baseline)

    def test_coverage_floor_and_budget_validation(self):
        for budgets in ({}, {"coarse": None}, {"coarse": (-1, 0, 0)}):
            with self.assertRaises(ValueError):
                admit_profiles(budgets, ["calibration"], [])
        with self.assertRaises(ValueError):
            admit_profiles({"detail": (0.001,) * 3}, ["calibration"], [], min_coverage=0)

    def test_stop_does_not_acquire(self):
        check = QualifiedViewConfirmation(0, self.admit())
        self.assertEqual(check.step(0, None, stop=True), "cancelled")
