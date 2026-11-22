import os
import signal
import time
import unittest

from embodied_agent.mock import MockPolicy
from embodied_agent.models import Action, Observation, Plan, PlanStep
from embodied_agent.process_witness import ProcessWitness, WitnessCancelled, WitnessTimeout
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


def crash(action):
    os._exit(23)


def crash_factory():
    return crash


def invalid(action):
    return None


def invalid_factory():
    return invalid


def fail(action):
    raise ValueError("private device details")


def fail_factory():
    return fail


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

    def test_worker_faults_stop_whole_task_without_reexecution(self):
        class Planner:
            def create_plan(self, instruction, observation):
                return Plan(
                    "verify", [PlanStep("verify", {"object": "b", "destination": "box"})] * 2
                )

        class Robot:
            calls = 0

            def observe(self):
                return read(None)

            def execute(self, action):
                self.calls += 1

        for factory_fn, code in (
            (crash_factory, "witness_disconnected"),
            (invalid_factory, "witness_read_failed"),
            (fail_factory, "witness_read_failed"),
        ):
            with self.subTest(code=code), ProcessWitness(factory_fn, timeout_s=1) as witness:
                robot = Robot()
                result = AgentRuntime(
                    Planner(),
                    MockPolicy(),
                    robot,
                    confirmation_timeout_s=2,
                    max_observation_age_s=1,
                    confirmation_witness=witness,
                ).run("verify")
                self.assertFalse(result.success)
                self.assertEqual(result.completed_steps, 0)
                self.assertEqual(robot.calls, 1)
                self.assertEqual(result.confirmation_reports[0]["status"], "witness_failed")
                self.assertEqual(result.confirmation_reports[0]["witness_failure"], code)
                self.assertIsNotNone(witness.exitcode)
                self.assertNotIn("private device", str(result))

    def test_cancellation_before_send_closes_without_request(self):
        with ProcessWitness(factory) as witness:
            with self.assertRaises(WitnessCancelled):
                witness(Action("verify"), stopped=lambda: True)
            self.assertEqual(witness.calls, 0)
            self.assertIsNotNone(witness.exitcode)

    def test_task_cancels_during_blocked_witness(self):
        class Planner:
            def create_plan(self, instruction, observation):
                return Plan(
                    "verify", [PlanStep("verify", {"object": "b", "destination": "box"})] * 2
                )

        class Robot:
            calls = 0

            def observe(self):
                return read(None)

            def execute(self, action):
                self.calls += 1

        with ProcessWitness(blocked_factory, timeout_s=5) as witness:
            robot = Robot()
            # Arm cancellation only after the witness request was sent.
            armed = None

            def stopped():
                nonlocal armed
                if witness.calls and armed is None:
                    armed = time.monotonic()
                return armed is not None and time.monotonic() - armed >= 0.05

            result = AgentRuntime(
                Planner(),
                MockPolicy(),
                robot,
                confirmation_timeout_s=5,
                max_observation_age_s=1,
                confirmation_witness=witness,
            ).run("verify", stop_requested=stopped)
            self.assertFalse(result.success)
            self.assertEqual(robot.calls, 1)
            self.assertEqual(result.completed_steps, 0)
            report = result.confirmation_reports[0]
            self.assertEqual(report["status"], "cancelled")
            self.assertLess(report["elapsed_s"], 2)
            self.assertIsNotNone(witness.exitcode)

    def test_invalid_timeout_and_startup_failure(self):
        for timeout in (True, 0, float("nan"), 6):
            with self.assertRaises(ValueError):
                ProcessWitness(factory, timeout_s=timeout)
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            ProcessWitness(failed_factory)


if __name__ == "__main__":
    unittest.main()
