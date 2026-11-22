import unittest

from scripts.evaluate_confirmation_stress import cases, evaluate


class StressTests(unittest.TestCase):
    def test_frozen_plan_reproducible_and_fully_paired(self):
        plan = cases()
        self.assertEqual(plan, cases())
        self.assertEqual(len(plan), 160)
        rows = evaluate(plan)
        self.assertEqual(len(rows), 480)
        self.assertEqual(len({(r["kind"], r["seed"], r["profile"]) for r in rows}), 480)
        self.assertTrue(all(r["executions"] == 1 for r in rows))

    def test_persistent_false_sensor_exposes_all_profiles_limit(self):
        rows = evaluate([c for c in cases() if c["kind"] == "persistent_false"])
        self.assertTrue(all(r["false_accept"] for r in rows))

    def test_plan_not_mutated_by_running_profiles(self):
        plan = cases()
        evaluate(plan)
        self.assertEqual(plan, cases())


if __name__ == "__main__":
    unittest.main()
