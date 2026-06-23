import asyncio
import os
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from whisper_not import api as api_server


def _fake_result():
    segment = SimpleNamespace(
        seek=0,
        start=0.0,
        end=1.0,
        text=" Hello.",
        tokens=[1],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.01,
        words=None,
    )
    info = SimpleNamespace(
        language="en",
        language_probability=0.99,
        duration=1.0,
        duration_after_vad=1.0,
    )
    return iter([segment]), info


class WorkerThreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_batch_transcription_runs_outside_the_event_loop_thread(self):
        event_loop_thread = threading.get_ident()
        inference_thread = None

        class RecordingModel:
            def transcribe(self, path, **kwargs):
                nonlocal inference_thread
                inference_thread = threading.get_ident()
                return _fake_result()

        original_model = api_server._model
        api_server._model = RecordingModel()
        try:
            segments, info = await api_server._run_transcription(
                "audio.wav",
                None,
                None,
                0.0,
                "transcribe",
                False,
            )
        finally:
            api_server._model = original_model

        self.assertNotEqual(inference_thread, event_loop_thread)
        self.assertEqual([segment.text for segment in segments], [" Hello."])
        self.assertEqual(info.language, "en")

    async def test_batch_transcription_does_not_block_other_coroutines(self):
        started = threading.Event()

        class SlowModel:
            def transcribe(self, path, **kwargs):
                started.set()
                time.sleep(0.1)
                return _fake_result()

        original_model = api_server._model
        api_server._model = SlowModel()
        try:
            task = asyncio.create_task(
                api_server._run_transcription(
                    "audio.wav",
                    None,
                    None,
                    0.0,
                    "transcribe",
                    False,
                )
            )
            await asyncio.to_thread(started.wait, 1)
            before = time.monotonic()
            await asyncio.sleep(0.01)
            elapsed = time.monotonic() - before
            await task
        finally:
            api_server._model = original_model

        self.assertLess(elapsed, 0.05)

    async def test_concurrent_calls_remain_strictly_serial(self):
        active = 0
        maximum_active = 0
        state_lock = threading.Lock()

        class SerialCheckModel:
            def transcribe(self, path, **kwargs):
                nonlocal active, maximum_active
                with state_lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.05)
                with state_lock:
                    active -= 1
                return _fake_result()

        original_model = api_server._model
        api_server._model = SerialCheckModel()
        try:
            await asyncio.gather(
                api_server._run_transcription(
                    "first.wav", None, None, 0.0, "transcribe", False
                ),
                api_server._run_transcription(
                    "second.wav", None, None, 0.0, "transcribe", False
                ),
            )
        finally:
            api_server._model = original_model

        self.assertEqual(maximum_active, 1)


class AdmissionLimiterTests(unittest.IsolatedAsyncioTestCase):
    async def test_limiter_bounds_active_and_queued_requests(self):
        limiter = api_server.AudioRequestLimiter(
            max_active=1,
            max_queued=1,
            wait_timeout=1.0,
        )
        await limiter.acquire()
        queued = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)

        with self.assertRaises(api_server.RequestQueueFull):
            await limiter.acquire()

        await limiter.release()
        await asyncio.wait_for(queued, timeout=0.2)
        await limiter.release()
        self.assertEqual(limiter.active, 0)
        self.assertEqual(limiter.waiting, 0)

    async def test_limiter_times_out_waiting_requests(self):
        limiter = api_server.AudioRequestLimiter(
            max_active=1,
            max_queued=1,
            wait_timeout=0.01,
        )
        await limiter.acquire()

        with self.assertRaises(api_server.RequestQueueFull):
            await limiter.acquire()

        await limiter.release()

    async def test_cancelled_waiter_hands_free_slot_to_next_request(self):
        class PausingLimiter(api_server.AudioRequestLimiter):
            def __init__(self):
                super().__init__(max_active=1, max_queued=2, wait_timeout=1.0)
                self.first_waiter_ready = asyncio.Event()
                self.resume_first_waiter = asyncio.Event()
                self.pause_first_waiter = True

            async def _wait_for_available_slot(self):
                await super()._wait_for_available_slot()
                if self.pause_first_waiter:
                    self.pause_first_waiter = False
                    self.first_waiter_ready.set()
                    await self.resume_first_waiter.wait()

        limiter = PausingLimiter()
        await limiter.acquire()
        first = asyncio.create_task(limiter.acquire())
        second = asyncio.create_task(limiter.acquire())
        await asyncio.sleep(0)

        await limiter.release()
        await asyncio.wait_for(limiter.first_waiter_ready.wait(), timeout=0.2)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first

        await asyncio.wait_for(second, timeout=0.2)
        await limiter.release()
        self.assertEqual(limiter.active, 0)
        self.assertEqual(limiter.waiting, 0)


class RequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.original_model = api_server._model
        self.original_model_name = api_server._model_name
        api_server._model = SimpleNamespace(transcribe=lambda *args, **kwargs: _fake_result())
        api_server._model_name = "test-model"
        self.client = TestClient(api_server.app)

    def tearDown(self):
        api_server._model = self.original_model
        api_server._model_name = self.original_model_name

    def test_api_key_comparison_uses_constant_time_helper(self):
        with patch.dict(os.environ, {"WHISPER_API_KEY": "expected-key"}):
            with patch.object(
                api_server.hmac,
                "compare_digest",
                wraps=api_server.hmac.compare_digest,
            ) as compare_digest:
                response = self.client.get(
                    "/v1/models",
                    headers={"Authorization": "Bearer wrong-key"},
                )

        self.assertEqual(response.status_code, 401)
        compare_digest.assert_called_once_with("wrong-key", "expected-key")

    def test_valid_api_key_still_allows_audio_requests(self):
        with patch.dict(os.environ, {"WHISPER_API_KEY": "expected-key"}):
            response = self.client.post(
                "/v1/audio/transcriptions",
                headers={"Authorization": "Bearer expected-key"},
                data={"model": "whisper-1"},
                files={"file": ("voice.wav", b"audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Hello."})

    def test_temperature_above_one_is_rejected(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "temperature": "1.1"},
            files={"file": ("voice.wav", b"audio", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("temperature", response.json()["detail"].lower())

    def test_temperature_below_zero_is_rejected(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "temperature": "-0.1"},
            files={"file": ("voice.wav", b"audio", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("temperature", response.json()["detail"].lower())

    def test_stream_accepts_standard_truthy_boolean_value(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "stream": "1"},
            files={"file": ("voice.wav", b"audio", "audio/wav")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("transcript.text.done", response.text)

    def test_invalid_stream_boolean_is_rejected(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1", "stream": "sometimes"},
            files={"file": ("voice.wav", b"audio", "audio/wav")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("stream", response.json()["detail"].lower())


if __name__ == "__main__":
    unittest.main()
