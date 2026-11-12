"""Linux process tests: the supervisor survives cleanup traps in the owned worker tree."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from scripts.run_frozen_study import supervised_command


@unittest.skipUnless(os.name == "posix", "owned process-group verification requires Linux")
class SupervisorTests(unittest.TestCase):
    def check_deadline(self, ignore_term):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            child_pid = output / "child.pid"
            child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)"
            worker = ("import signal,subprocess,sys; from pathlib import Path\n" +
                      ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n" if ignore_term else "") +
                      f"with subprocess.Popen([sys.executable,'-c',{child!r}]) as proc:\n" +
                      f" Path({str(child_pid)!r}).write_text(str(proc.pid))\n" +
                      " raise TimeoutError('cleanup trap: Popen.__exit__ waits for 60 seconds')\n")
            receipt = supervised_command([sys.executable, "-c", worker], seconds=.6,
                                         output=output, grace_seconds=.2)
            self.assertTrue(receipt["timed_out"])
            self.assertLess(receipt["seconds"], 3)
            self.assertNotEqual(receipt["returncode"], 0)
            self.assertTrue(child_pid.exists())
            # A killed orphan may briefly be a zombie awaiting init; it cannot compute.
            status = Path("/proc") / child_pid.read_text() / "stat"
            deadline = time.monotonic() + .5
            while status.exists() and status.read_text().split()[2] != "Z" and time.monotonic() < deadline:
                time.sleep(.01)  # SIGKILL delivery precedes the kernel's exit-state transition.
            if status.exists():
                self.assertEqual(status.read_text().split()[2], "Z")
            print(json.dumps(dict(ignore_term=ignore_term, **receipt)))

    def test_context_manager_wait_does_not_defeat_deadline(self):
        self.check_deadline(False)

    def test_ignored_termination_is_escalated_only_for_owned_group(self):
        self.check_deadline(True)

    def test_normal_worker_retains_success(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = supervised_command([sys.executable, "-c", "print('finished')"],
                                         seconds=3, output=Path(folder))
            self.assertFalse(receipt["timed_out"])
            self.assertEqual(receipt["returncode"], 0)
            self.assertEqual((Path(folder)/"worker.stdout").read_text().strip(), "finished")


if __name__ == "__main__":
    unittest.main()
