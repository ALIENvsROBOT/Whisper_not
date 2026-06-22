import pathlib
import unittest


class StartupTimeoutRegressionTests(unittest.TestCase):
    def test_model_download_timeout_is_configurable_and_defaults_to_one_hour(self):
        script = pathlib.Path("run.sh").read_text(encoding="utf-8")

        self.assertIn(
            "[ -z \"$WHISPER_STARTUP_TIMEOUT\" ] && WHISPER_STARTUP_TIMEOUT=3600",
            script,
        )
        self.assertIn('while [ "$i" -lt "$WHISPER_STARTUP_TIMEOUT" ]', script)
        self.assertNotIn('while [ "$i" -lt 300 ]', script)


if __name__ == "__main__":
    unittest.main()
