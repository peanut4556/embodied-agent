import unittest

from scripts.evaluate_confirmation_stress import evaluate as single_evaluate
from scripts.evaluate_witness_stress import evaluate, plan


class WitnessStressTests(unittest.TestCase):
    def test_plan_reproducible_and_runs_paired(self):
        campaign = plan()
        self.assertEqual(campaign, plan())
        rows = evaluate(campaign)
        self.assertEqual(len(rows), 640)
        self.assertEqual(len({(r["mode"], r["kind"], r["seed"]) for r in rows}), 640)
        self.assertTrue(all(r["executions"] == 1 for r in rows))
        self.assertEqual(campaign, plan())

    def test_common_mode_does_not_solve_persistent_false(self):
        rows = evaluate([c for c in plan() if c["kind"] == "persistent_false"])
        self.assertTrue(all(r["false_accept"] for r in rows if r["mode"] == "common_mode"))
        self.assertFalse(any(r["false_accept"] for r in rows if r["mode"] == "ideal_reference"))

    def test_single_source_matches_previous_campaign(self):
        campaign = plan()
        old = {
            (r["kind"], r["seed"]): r["success"]
            for r in single_evaluate(campaign)
            if r["profile"] == "temporal"
        }
        new = {
            (r["kind"], r["seed"]): r["success"]
            for r in evaluate(campaign)
            if r["mode"] == "single"
        }
        self.assertEqual(old, new)


if __name__ == "__main__":
    unittest.main()
