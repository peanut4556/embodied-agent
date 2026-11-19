import multiprocessing
import tempfile
import unittest
from pathlib import Path

from embodied_agent.wait_budget_store import WaitBudgetStore


def compete(path, checkpoint, identity, gate, reports):
    gate.wait(5)
    try:
        WaitBudgetStore(path).reserve(identity, reviewed_checkpoint=checkpoint)
    except RuntimeError as exc:
        reports.put(str(exc))
    else:
        reports.put("reserved")


class ReviewedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "pool.db"
        self.store = WaitBudgetStore.create(self.path, total_threshold_ns=10, slots=3)

    def test_stale_review_refused_even_with_remaining_capacity(self):
        review = self.store.checkpoint()
        self.store.reserve("other")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            self.store.reserve("reviewed", reviewed_checkpoint=review)
        self.assertEqual(self.store.snapshot()["reserved_slots"], 1)
        result = self.store.reserve("reviewed", reviewed_checkpoint=self.store.checkpoint())
        self.assertEqual(result["slot"], 2)

    def test_prefix_binding_and_exact_review_have_distinct_semantics(self):
        baseline = self.store.checkpoint()
        pinned = WaitBudgetStore(self.path, expected_checkpoint=baseline)
        pinned.reserve("a")
        pinned.reserve("b")
        with self.assertRaisesRegex(RuntimeError, "stale"):
            pinned.reserve("c", reviewed_checkpoint=baseline)

    def test_invalid_review_does_not_allocate(self):
        for review in ({}, self.store.checkpoint() | {"reserved_slots": True}):
            with self.assertRaises(ValueError):
                self.store.reserve("a", reviewed_checkpoint=review)
        self.assertEqual(self.store.snapshot()["reserved_slots"], 0)

    def test_two_processes_cannot_both_spend_same_review(self):
        ctx = multiprocessing.get_context("spawn")
        gate, reports = ctx.Event(), ctx.Queue()
        review = self.store.checkpoint()
        children = [
            ctx.Process(target=compete, args=(self.path, review, str(i), gate, reports))
            for i in range(2)
        ]
        try:
            for child in children:
                child.start()
            gate.set()
            results = [reports.get(timeout=10) for _ in children]
            for child in children:
                child.join(5)
                self.assertEqual(child.exitcode, 0)
            self.assertCountEqual(results, ["reserved", "reviewed budget checkpoint is stale"])
            self.assertEqual(self.store.snapshot()["reserved_slots"], 1)
        finally:
            for child in children:
                if child.pid is not None:
                    if child.is_alive():
                        child.kill()
                    child.join(5)
                    child.close()
            reports.close()
            reports.join_thread()


if __name__ == "__main__":
    unittest.main()
