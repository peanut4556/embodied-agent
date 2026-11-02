import unittest

from test_process_post_confirmation import Session

from embodied_agent.exposure_confirmation import ExposureClockMap, ExposureConfirmation
from embodied_agent.process_exposure_observer import ProcessExposureObserver
from embodied_agent.process_post_confirmation import ProcessPostConfirmation
from embodied_agent.recovery_exposure_observer import RecoveryExposureObserver


class OwnershipTests(unittest.TestCase):
    def observer(self, recovery=False):
        session = Session()
        observer = (
            RecoveryExposureObserver(session, 0, task_clock=lambda: 1)
            if recovery
            else ProcessExposureObserver(session, 0)
        )
        # A terminal-state fixture; the integration suite obtains this via real capture.
        observer.confirmation = ExposureConfirmation(0, ExposureClockMap(16, 0))
        observer.status, observer.last_task = "confirmed_in_tray", 1
        return observer

    def test_duplicate_handoff_and_direct_claim_are_rejected(self):
        observer = self.observer()
        followup = ProcessPostConfirmation.from_confirmed(observer)
        with self.assertRaises(RuntimeError):
            ProcessPostConfirmation.from_confirmed(observer)
        with self.assertRaises(RuntimeError):
            ProcessPostConfirmation(observer.session, observer.result(), 1, ExposureClockMap(16, 0))
        self.assertEqual(observer.session.requests, 0)
        followup.close()

    def test_original_close_cannot_close_transferred_session_or_change_result(self):
        for recovery in (False, True):
            observer = self.observer(recovery)
            original = observer.result()
            followup = ProcessPostConfirmation.from_confirmed(observer)
            observer.close()
            self.assertEqual(observer.session.status, "ready")
            self.assertEqual(observer.result(), original)
            if recovery:
                self.assertEqual(observer.cleanup["status"], "not_requested")
            followup.close()
            self.assertEqual(observer.session.status, "closed")

    def test_failed_construction_does_not_consume_handoff(self):
        observer = self.observer()
        with self.assertRaises(ValueError):
            ProcessPostConfirmation.from_confirmed(observer, duration=0)
        self.assertFalse(observer.session_transferred)
        followup = ProcessPostConfirmation.from_confirmed(observer)
        followup.close()

    def test_replaced_session_cannot_be_closed_by_old_owner(self):
        observer = self.observer()
        old = ProcessPostConfirmation.from_confirmed(observer)
        observer.session.session = "replacement"
        new = ProcessPostConfirmation(
            observer.session, observer.result(), 2, ExposureClockMap(16, 0)
        )
        self.assertEqual(old.step(1.25), "invalidated")
        self.assertEqual(observer.session.polls, 0)
        with self.assertRaises(RuntimeError):
            old.close()
        self.assertEqual(observer.session.status, "ready")
        new.close()
        self.assertEqual(observer.session.status, "closed")

    def test_context_preserves_body_and_cleanup_errors(self):
        observer = self.observer()
        followup = ProcessPostConfirmation.from_confirmed(observer)
        observer.session.fail_close = True
        with self.assertRaises(ExceptionGroup) as raised, followup:
            raise ValueError("body failed")
        self.assertEqual(
            [str(e) for e in raised.exception.exceptions], ["body failed", "cleanup failed"]
        )
        self.assertEqual(followup.result()["cleanup"]["status"], "failed")
        observer.session.fail_close = False
        followup.close()
        self.assertFalse(followup.result()["cleanup_required"])

    def test_successful_cleanup_does_not_suppress_body_exception(self):
        followup = ProcessPostConfirmation.from_confirmed(self.observer())
        with self.assertRaisesRegex(ValueError, "body failed"), followup:
            raise ValueError("body failed")
        self.assertEqual(followup.result()["cleanup"]["status"], "succeeded")

    def test_closed_adapter_is_idempotent_even_after_new_session(self):
        observer = self.observer()
        old = ProcessPostConfirmation.from_confirmed(observer)
        old.close()
        observer.session.session, observer.session.status = "new", "ready"
        old.close()
        old.step(2)
        self.assertEqual(observer.session.status, "ready")
        self.assertEqual(observer.session.polls, 0)
