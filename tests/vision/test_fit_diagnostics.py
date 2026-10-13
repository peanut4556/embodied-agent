"""Instrumented replay must preserve the original solver's numerical behavior."""

import json
import unittest
from pathlib import Path

import numpy as np

from embodied_agent.reference_fit import fit_reference, jacobian, predict
from embodied_agent.reference_fit_diagnostics import finite_difference_jacobian, trace_reference_fit
from scripts.audit_exposure_batches import draw
from scripts.audit_fit_stalls import select_cases, summarize


class FitDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        baseline = json.loads(Path("docs/evaluations/redundant-reference-v1.json").read_text())
        self.planes = [(np.array(p["normal"]), p["offset_m"]) for p in baseline["planes"]]
        self.truth = np.array(baseline["true_parameters"])
        self.rays = np.array(
            [
                [x, y, -1.0]
                for x in np.linspace(-0.025, 0.025, 21)
                for y in np.linspace(-0.02, 0.02, 21)
            ]
        )

    def test_stalled_replay_matches_every_original_field(self):
        clean = predict(self.rays, self.planes, self.truth)
        observed = draw(clean, 50003, 0, 0)[:3]
        expected = fit_reference(self.rays, self.planes[:3], observed)
        traced = trace_reference_fit(self.rays, self.planes[:3], observed)
        self.assertEqual(expected, traced["fit"])
        self.assertEqual(expected["status"], "stalled")
        summary = summarize(
            [{"group": "narrow_stalled", "noise_model": "independent", "diagnostic": traced}]
        )
        json.dumps({"trace": traced, "summary": summary}, allow_nan=False)
        self.assertEqual(len(traced["trace"][-1]["trials"]), 12)
        self.assertTrue(all(t["outcome"] != "accepted" for t in traced["trace"][-1]["trials"]))

    def test_noiseless_and_rank_deficient_replays(self):
        for planes in (self.planes[:3], self.planes[:1]):
            observed = predict(self.rays, planes, self.truth)
            original = fit_reference(self.rays, planes, observed)
            traced = trace_reference_fit(self.rays, planes, observed)
            self.assertEqual(traced["fit"], original)
            self.assertEqual(len(traced["trace"]), original["iterations"])

    def test_finite_difference_probe_matches_baseline(self):
        np.testing.assert_array_equal(
            finite_difference_jacobian(self.rays, self.planes[:3], self.truth, 1e-6),
            jacobian(self.rays, self.planes[:3], self.truth),
        )
        with self.assertRaises(ValueError):
            finite_difference_jacobian(self.rays, self.planes[:3], self.truth, 0)
        with self.assertRaises(ValueError):
            trace_reference_fit(self.rays, self.planes[:3], np.zeros((1, 2)))

    def test_selection_preserves_pairing_and_control_limit(self):
        conditions = []
        for patch in ("wide", "narrow"):
            for model in ("independent", "global_shared", "exposure_shared"):
                cases = [
                    {
                        "seed": s,
                        "fit_status": "stalled" if patch == "narrow" and s == 1 else "converged",
                    }
                    for s in (1, 2, 3)
                ]
                conditions.append(
                    {"patch": patch, "noise_model": model, "batches": [{"cases": cases}]}
                )
        selected = select_cases({"conditions": conditions}, 1)
        self.assertEqual(len(selected), 9)
        for model in ("independent", "global_shared", "exposure_shared"):
            rows = [c for c in selected if c["noise_model"] == model]
            self.assertEqual(
                [(c["group"], c["seed"]) for c in rows],
                [("narrow_stalled", 1), ("wide_paired", 1), ("narrow_converged_control", 2)],
            )
