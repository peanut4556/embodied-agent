import time
import unittest

from scripts.check_wait_under_load import BoundedLoad


class LoadTests(unittest.TestCase):
    def test_context_stops_workers_on_exception(self):
        load = BoundedLoad(count=1)
        with self.assertRaisesRegex(RuntimeError, "test"), load:
            raise RuntimeError("test")
        self.assertEqual(load.exitcodes, [0])
        self.assertEqual(load.events[-1]["reason"], "requested")

    def test_lifetime_ends_without_stop_signal(self):
        load = BoundedLoad(count=1, lifetime=0.2)
        with load:
            time.sleep(0.4)
        self.assertEqual(load.exitcodes, [0])
        self.assertEqual(load.events[-1]["reason"], "lifetime")
        self.assertGreater(load.events[-1]["cpu_s"], 0)

    def test_invalid_resource_bounds(self):
        for kwargs in (
            {"count": 0},
            {"count": 5},
            {"count": True},
            {"lifetime": float("nan")},
            {"lifetime": 31},
        ):
            with self.assertRaises(ValueError):
                BoundedLoad(**kwargs)


if __name__ == "__main__":
    unittest.main()
