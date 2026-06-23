import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from whisper_not import api as api_server


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(kwargs)
        segment = SimpleNamespace(
            seek=0,
            start=0.0,
            end=1.0,
            text=" Hello world.",
            tokens=[1, 2],
            temperature=0.0,
            avg_logprob=-0.1,
            compression_ratio=1.0,
            no_speech_prob=0.01,
            words=[
                SimpleNamespace(word=" Hello", start=0.0, end=0.5, probability=0.9),
                SimpleNamespace(word=" world", start=0.5, end=1.0, probability=0.95),
            ],
        )
        info = SimpleNamespace(
            language="en",
            language_probability=0.99,
            duration=1.0,
            duration_after_vad=1.0,
        )
        return iter([segment]), info


class OpenWebUICompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.original_model = api_server._model
        self.original_model_name = api_server._model_name
        self.original_mode = api_server._diarization_mode
        self.original_diarizer = api_server.diarization
        self.model = FakeModel()
        api_server._model = self.model
        api_server._model_name = "test-model"
        api_server._diarization_mode = "on_demand"
        self.client = TestClient(api_server.app)

    def tearDown(self):
        api_server._model = self.original_model
        api_server._model_name = self.original_model_name
        api_server._diarization_mode = self.original_mode
        api_server.diarization = self.original_diarizer

    def test_openwebui_default_request_returns_plain_json_without_extra_work(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files={"file": ("voice.webm", b"audio", "audio/webm")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "Hello world."})
        self.assertFalse(self.model.calls[0]["word_timestamps"])

    def test_models_endpoint_exposes_openwebui_compatible_alias(self):
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        model_ids = {item["id"] for item in response.json()["data"]}
        self.assertIn("whisper-1", model_ids)
        self.assertIn("test-model", model_ids)

    def test_direct_request_enables_word_timestamps_only_when_requested(self):
        response = self.client.post(
            "/v1/audio/transcriptions",
            files=[
                ("model", (None, "whisper-1")),
                ("response_format", (None, "verbose_json")),
                ("timestamp_granularities[]", (None, "word")),
                ("file", ("voice.wav", b"audio", "audio/wav")),
            ],
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.model.calls[0]["word_timestamps"])
        self.assertEqual(response.json()["words"][0]["word"], "Hello")

    def test_direct_diarization_request_runs_local_diarizer(self):
        call_order = []
        fake_diarizer = SimpleNamespace(
            assign_speakers=lambda segments, turns: [
                segment.update({"speaker": "SPEAKER_00"}) or segment
                for segment in segments
            ],
        )
        fake_diarizer.resolve_provider = lambda *args, **kwargs: "cpu"
        fake_worker = SimpleNamespace(
            load=lambda **kwargs: call_order.append("diarizer_load"),
            diarize=lambda *args, **kwargs: (
                call_order.append("diarize")
                or [(0.0, 1.0, "SPEAKER_00")]
            ),
        )
        api_server.diarization = fake_diarizer
        original_transcribe = self.model.transcribe

        def record_transcribe(*args, **kwargs):
            call_order.append("transcribe")
            return original_transcribe(*args, **kwargs)

        self.model.transcribe = record_transcribe

        with patch.object(api_server, "_diarization_process", fake_worker):
            response = self.client.post(
                "/v1/audio/transcriptions",
                data={
                    "model": "gpt-4o-transcribe-diarize",
                    "response_format": "diarized_json",
                    "download": "true",
                },
                files={"file": ("meeting.wav", b"audio", "audio/wav")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["segments"][0]["speaker"], "A")
        self.assertIn("meeting.json", response.headers["content-disposition"])
        self.assertEqual(
            call_order,
            ["diarizer_load", "transcribe", "diarize"],
        )


if __name__ == "__main__":
    unittest.main()
