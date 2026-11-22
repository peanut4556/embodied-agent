import unittest

from embodied_agent.models import Action, Observation
from embodied_agent.runtime import AgentRuntime


class WitnessTests(unittest.TestCase):
    def confirm(self, witness, samples=1):
        self.t = 10.0
        test = self

        class Robot:
            def observe(self):
                return Observation(objects={"b": "box"}, metadata={"observed_at_monotonic": test.t})

        runtime = AgentRuntime(
            None,
            None,
            Robot(),
            confirmation_timeout_s=0.05,
            max_observation_age_s=0.1,
            confirmation_samples=samples,
            confirmation_witness=witness,
            max_witness_skew_s=0.02,
            clock=lambda: self.t,
            sleep=lambda d: setattr(self, "t", self.t + d),
        )
        report = {}
        status = runtime._await_confirmation(
            Action("verify", {"object": "b", "destination": "box"}), lambda: False, report=report
        )
        return status, report

    def observation(self, positive=True, stamp=None):
        return Observation(
            objects={"b": "box" if positive else "table"},
            metadata={"observed_at_monotonic": self.t if stamp is None else stamp},
        )

    def test_persistent_primary_false_positive_blocked_by_witness(self):
        status, report = self.confirm(lambda _: self.observation(False))
        self.assertEqual(status, "timeout")
        self.assertGreater(report["witness_disagreements"], 0)

    def test_agreement_can_confirm_multiple_samples(self):
        self.assertEqual(self.confirm(lambda _: self.observation(), samples=3)[0], "confirmed")

    def test_cached_witness_cannot_accumulate(self):
        self.assertEqual(
            self.confirm(lambda _: self.observation(stamp=10), samples=3)[0], "timeout"
        )

    def test_stale_future_and_invalid_witness_fail(self):
        self.assertEqual(self.confirm(lambda _: self.observation(stamp=9))[0], "timeout")
        self.assertEqual(
            self.confirm(lambda _: self.observation(stamp=11))[0], "invalid_witness_timestamp"
        )
        self.assertEqual(self.confirm(lambda _: None)[0], "invalid_witness_observation")

    def test_configuration_requires_freshness(self):
        with self.assertRaises(ValueError):
            AgentRuntime(None, None, None, confirmation_witness=lambda _: None)


if __name__ == "__main__":
    unittest.main()
