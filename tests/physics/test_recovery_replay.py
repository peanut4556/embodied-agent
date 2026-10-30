import copy
import json
import unittest
from pathlib import Path

from embodied_agent.recovery_audit import validate_report
from embodied_agent.recovery_replay import replay_report
from scripts.check_recovery_reports import scenarios


class RecoveryReplayTests(unittest.TestCase):
    def setUp(self):
        self.cases = scenarios()["cases"]
        self.confirmed = self.cases["confirmed"]["after_close"]

    def observations(self, report):
        return [row for row in report["events"] if row["event"] == "observation"]

    def test_replay_all_new_scenarios_and_json_round_trip(self):
        for case in self.cases.values():
            for stage in ("before_close", "after_close"):
                report = json.loads(json.dumps(case[stage]))
                original = copy.deepcopy(report)
                replay = validate_report(report)["evidence_replay"]
                self.assertEqual(replay["status"], "verified")
                self.assertEqual(
                    replay["visual_confirmation_reproduced"],
                    report["diagnosis"]["visual_confirmation_succeeded"],
                )
                self.assertEqual(report, original)
        replay = replay_report(self.confirmed)
        self.assertEqual(replay["observations_replayed"], 6)

    def test_changed_position_breaks_claimed_stability(self):
        self.observations(self.confirmed)[0]["position"][0] = 0.8
        with self.assertRaisesRegex(ValueError, "differs|not reproduced"):
            replay_report(self.confirmed)

    def test_deleted_frame_cannot_bridge_exposure_gap(self):
        frame = self.observations(self.confirmed)[2]
        self.confirmed["events"].remove(frame)
        with self.assertRaisesRegex(ValueError, "differs|not reproduced"):
            replay_report(self.confirmed)

    def test_changed_timestamp_mapping_and_session_are_rejected(self):
        for kind in ("timestamp", "mapping", "session", "missing", "task_order"):
            report = copy.deepcopy(self.confirmed)
            rows = self.observations(report)
            if kind == "timestamp":
                rows[0]["captured_at"] += 1
            elif kind == "mapping":
                next(row for row in report["events"] if row["event"] == "mapping")["mapping"][
                    "rate"
                ] = 2
            elif kind == "session":
                rows[0]["session"] = "old"
            elif kind == "task_order":
                rows[1]["task_time"] = rows[0]["task_time"]
            else:
                del rows[0]["position"]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                replay_report(report)

    def test_forged_success_with_consistent_summary_is_rejected(self):
        report = self.cases["restored_pending"]["before_close"]
        report["result"]["status"] = "confirmed_in_tray"
        report["result"]["reason"] = "stable_visual_evidence"
        report["diagnosis"].update(
            visual_confirmation_succeeded=True, terminal=True, reason_code="stable_visual_evidence"
        )
        with self.assertRaisesRegex(ValueError, "not reproduced"):
            validate_report(report)

    def test_reset_requires_new_mapping_and_terminal_blocks_frames(self):
        for event in ("awaiting_clock_mapping", "user_stop"):
            report = copy.deepcopy(self.confirmed)
            index = report["events"].index(self.observations(report)[3])
            report["events"].insert(index, {"event": event, "session": "replacement"})
            with self.assertRaises(ValueError):
                replay_report(report)

    def test_legacy_reports_are_explicitly_unavailable(self):
        root = Path(__file__).resolve().parents[2]
        legacy = json.loads((root / "docs/evaluations/recovery-report-v1.json").read_text())
        for case in legacy["cases"].values():
            replay = validate_report(case["after_close"])["evidence_replay"]
            self.assertEqual(replay["status"], "unavailable")

    def test_unknown_version_and_truncated_evidence_fail_closed(self):
        for evidence in ({"schema_version": 2, "start_time": 0}, {}, None):
            report = copy.deepcopy(self.confirmed)
            report["evidence"] = evidence
            with self.assertRaises(ValueError):
                replay_report(report)
        self.confirmed["events"] = []
        with self.assertRaises(ValueError):
            replay_report(self.confirmed)
