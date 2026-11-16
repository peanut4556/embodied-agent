import multiprocessing
import time
import unittest

from embodied_agent.capture_supervisor import _run_supervised


def blocked():
    while True:
        time.sleep(0.01)


class SupervisorLifelineTests(unittest.TestCase):
    def test_lifeline_loss_exits_blocked_child(self):
        ctx = multiprocessing.get_context("spawn")
        reader, writer = ctx.Pipe(duplex=False)
        child = ctx.Process(target=_run_supervised, args=(blocked, reader))
        child.start()
        reader.close()
        try:
            writer.close()
            child.join(2)
            self.assertEqual(child.exitcode, 87)
        finally:
            if child.is_alive():
                child.kill()
                child.join(1)
            child.close()


if __name__ == "__main__":
    unittest.main()
