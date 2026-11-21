import unittest

from scripts.evaluate_confirmation_faults import evaluate


class MatrixTests(unittest.TestCase):
    def test_matrix_is_complete_and_never_reexecutes(self):
        rows = evaluate()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({(r["profile"], r["case"]) for r in rows}), 18)
        self.assertTrue(all(r["executions"] == 1 for r in rows))

    def test_temporal_rejects_scripted_faults_and_accepts_valid_updates(self):
        rows = [r for r in evaluate() if r["profile"] == "temporal"]
        self.assertTrue(all(r["success"] == r["expected_success"] for r in rows))
        self.assertTrue(any(r["false_accept"] for r in evaluate() if r["profile"] == "immediate"))

    def test_repeated_evaluation_has_no_cross_case_state(self):
        self.assertEqual(evaluate(), evaluate())


if __name__ == "__main__":
    unittest.main()
