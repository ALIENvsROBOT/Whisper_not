import unittest

from whisper_not.diarization_process import DiarizationProcess


class DiarizationProcessTests(unittest.TestCase):
    def test_spawned_worker_round_trip_and_shutdown(self):
        worker = DiarizationProcess(request_timeout=10)
        try:
            self.assertEqual(worker.ping(), "pong")
            self.assertTrue(worker._process.is_alive())
        finally:
            worker.close()

        self.assertIsNone(worker._process)
        self.assertIsNone(worker._connection)

    def test_timeout_must_be_positive(self):
        with self.assertRaises(ValueError):
            DiarizationProcess(request_timeout=0)


if __name__ == "__main__":
    unittest.main()
