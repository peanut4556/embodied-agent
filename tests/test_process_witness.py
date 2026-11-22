import signal
import time
import unittest

from embodied_agent.models import Action, Observation
from embodied_agent.process_witness import ProcessWitness, WitnessTimeout
from embodied_agent.runtime import AgentRuntime


def read(action):
    return Observation(objects={"b": "box"}, metadata={"observed_at_monotonic": time.monotonic()})


def factory():
    return read


def blocked(action):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    while True:
        time.sleep(0.01)


def blocked_factory():
    return blocked


def failed_factory():
    raise RuntimeError("failed")


class WitnessProcessTests(unittest.TestCase):
    def test_real_observation_and_idempotent_close(self):
        with ProcessWitness(factory) as witness:
            value = witness(Action("verify"))
            self.assertEqual(value.objects["b"], "box")
        self.assertTrue(witness.close())
        self.assertIsNotNone(witness.exitcode)

    def test_blocked_callback_killed_and_cannot_be_replayed(self):
        witness = ProcessWitness(blocked_factory, timeout_s=0.03)
        try:
            with self.assertRaises(WitnessTimeout):
                witness(Action("verify"))
            self.assertIsNotNone(witness.exitcode)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                witness(Action("verify"))
            self.assertEqual(witness.calls, 1)
        finally:
            witness.close()

    def test_runtime_reports_witness_timeout_without_single_source_success(self):
        class Robot:
            def observe(self):
                return read(None)

        with ProcessWitness(blocked_factory, timeout_s=0.03) as witness:
            runtime = AgentRuntime(
                None,
                None,
                Robot(),
                confirmation_timeout_s=0.5,
                max_observation_age_s=0.1,
                confirmation_witness=witness,
            )
            report = {}
            status = runtime._await_confirmation(
                Action("verify", {"object": "b", "destination": "box"}),
                lambda: False,
                report=report,
            )
            self.assertEqual(status, "witness_timeout")
            self.assertEqual(report["status"], "witness_timeout")
            self.assertGreaterEqual(report["elapsed_s"], 0.03)
            self.assertEqual(report["witness_reads"], 0)

    def test_invalid_timeout_and_startup_failure(self):
        for timeout in (True, 0, float("nan"), 6):
            with self.assertRaises(ValueError):
                ProcessWitness(factory, timeout_s=timeout)
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            ProcessWitness(failed_factory)


if __name__ == "__main__":
    unittest.main()
