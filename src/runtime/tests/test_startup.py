"""Test startup cancellation against a blocked CUDA stub, with no GPU or DDS."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest


BINARY = STUB = MODELS = None


class StartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if BINARY is None:
            raise unittest.SkipTest("Run through CTest with the runtime and CUDA stub paths")

    def start_runtime(self, no_gpu=False):
        temp = tempfile.TemporaryDirectory(prefix="sonic-startup-")
        self.addCleanup(temp.cleanup)
        path = Path(temp.name)
        log = (path / "runtime.log").open("w+")
        self.addCleanup(log.close)
        env = dict(os.environ, LD_PRELOAD=STUB)
        env.pop("SONIC_TEST_NO_GPU", None)
        if no_gpu:
            env["SONIC_TEST_NO_GPU"] = "1"
        args = [BINARY, "--dry-run", "--model-dir", MODELS]
        for option in ("command", "state", "head", "hands"):
            args += [f"--{option}-bind", f"ipc://{path}/{option}"]
        process = subprocess.Popen(args, env=env, stdout=log, stderr=log,
                                   start_new_session=True)

        def stop():
            if process.poll() is None:
                process.kill()
            process.wait(timeout=2)

        self.addCleanup(stop)
        return process, log

    @staticmethod
    def output(log):
        log.seek(0)
        return log.read()

    def check_cancel(self, signum):
        process, log = self.start_runtime()
        deadline = time.monotonic() + 10
        while "TEST: CUDA initialization is blocked" not in self.output(log):
            self.assertIsNone(process.poll(), self.output(log))
            self.assertLess(time.monotonic(), deadline, self.output(log))
            time.sleep(.02)
        self.assertIn("Initializing CUDA device", self.output(log))
        self.assertNotIn("Policy models ready", self.output(log))
        process.send_signal(signum)
        self.assertEqual(process.wait(timeout=2), -signum)

    def test_sigint_cancels_model_initialization(self):
        self.check_cancel(signal.SIGINT)

    def test_sigterm_cancels_model_initialization(self):
        self.check_cancel(signal.SIGTERM)

    def test_missing_gpu_exits_without_building_engine(self):
        process, log = self.start_runtime(no_gpu=True)
        self.assertEqual(process.wait(timeout=10), 1)
        self.assertIn("CUDA device index is unavailable", self.output(log))
        self.assertNotIn("building engine", self.output(log))
        self.assertNotIn("Policy models ready", self.output(log))


if __name__ == "__main__":
    BINARY, STUB, MODELS = map(lambda p: str(Path(p).resolve()), sys.argv[1:4])
    del sys.argv[1:4]
    unittest.main()
