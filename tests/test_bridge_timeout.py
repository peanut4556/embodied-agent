import io
import unittest
from unittest.mock import patch

from embodied_agent.sim_app import bridge


class BridgeTimeoutTests(unittest.TestCase):
    def test_read_and_stop_are_short_without_shortening_motion_budget(self):
        for payload, expected in [
            (None, 3),
            ({"name": "stop"}, 3),
            ({"name": "pick"}, 25),
            ({"name": "reset"}, 25),
            ({"name": "learned_pick_place"}, 95),
        ]:
            with (
                self.subTest(payload=payload),
                patch(
                    "embodied_agent.sim_app.urlopen", return_value=io.BytesIO(b'{"success":true}')
                ) as opened,
            ):
                self.assertTrue(bridge(payload)["success"])
                self.assertEqual(opened.call_args.kwargs["timeout"], expected)
                opened.assert_called_once()

    def test_timeout_propagates_without_automatic_command_replay(self):
        with patch(
            "embodied_agent.sim_app.urlopen", side_effect=TimeoutError("timed out")
        ) as opened:
            with self.assertRaises(TimeoutError):
                bridge({"name": "stop", "run_id": "test-run"})
            opened.assert_called_once()


if __name__ == "__main__":
    unittest.main()
