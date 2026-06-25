import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

from whisper_not import api as api_server


class ResolveRequestOptionsTests(unittest.TestCase):
    def setUp(self):
        self.original_mode = api_server._diarization_mode
        api_server._diarization_mode = "on_demand"

    def tearDown(self):
        api_server._diarization_mode = self.original_mode

    def test_default_openwebui_request_keeps_optional_work_disabled(self):
        options = api_server._resolve_request_options(
            model="whisper-1",
            response_format="json",
            timestamp_granularities=None,
            diarize=None,
            num_speakers=None,
            diarize_threshold=None,
            download=None,
        )

        self.assertFalse(options.word_timestamps)
        self.assertFalse(options.diarize)
        self.assertFalse(options.download)
        self.assertEqual(options.response_format, "json")

    def test_explicit_flags_enable_optional_features(self):
        options = api_server._resolve_request_options(
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
            diarize="true",
            num_speakers=2,
            diarize_threshold=0.42,
            download="true",
        )

        self.assertTrue(options.word_timestamps)
        self.assertTrue(options.diarize)
        self.assertTrue(options.download)
        self.assertEqual(options.num_speakers, 2)
        self.assertEqual(options.diarize_threshold, 0.42)

    def test_openai_diarization_model_maps_to_diarized_json(self):
        options = api_server._resolve_request_options(
            model="gpt-4o-transcribe-diarize",
            response_format="json",
            timestamp_granularities=None,
            diarize=None,
            num_speakers=None,
            diarize_threshold=None,
            download=None,
        )

        self.assertTrue(options.diarize)
        self.assertEqual(options.response_format, "diarized_json")

    def test_diarized_json_enables_diarization(self):
        options = api_server._resolve_request_options(
            model="whisper-1",
            response_format="diarized_json",
            timestamp_granularities=None,
            diarize=None,
            num_speakers=None,
            diarize_threshold=None,
            download=None,
        )

        self.assertTrue(options.diarize)

    def test_disabled_diarization_rejects_requested_diarization(self):
        api_server._diarization_mode = "disabled"

        with self.assertRaises(HTTPException) as error:
            api_server._resolve_request_options(
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=None,
                diarize="true",
                num_speakers=None,
                diarize_threshold=None,
                download=None,
            )

        self.assertEqual(error.exception.status_code, 400)

    def test_invalid_speaker_count_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            api_server._resolve_request_options(
                model="whisper-1",
                response_format="verbose_json",
                timestamp_granularities=None,
                diarize="true",
                num_speakers=0,
                diarize_threshold=None,
                download=None,
            )

        self.assertEqual(error.exception.status_code, 400)

    def test_word_granularity_requires_verbose_json(self):
        with self.assertRaises(HTTPException) as error:
            api_server._resolve_request_options(
                model="whisper-1",
                response_format="json",
                timestamp_granularities=["word"],
                diarize=None,
                num_speakers=None,
                diarize_threshold=None,
                download=None,
            )

        self.assertEqual(error.exception.status_code, 400)


class TranscriptionOptionTests(unittest.TestCase):
    def test_transcription_uses_long_audio_anti_loop_defaults(self):
        calls = []

        class Model:
            def transcribe(self, *args, **kwargs):
                calls.append(kwargs)
                segment = SimpleNamespace(text="Hello.")
                info = SimpleNamespace(language="en")
                return iter([segment]), info

        original_model = api_server._model
        api_server._model = Model()
        try:
            with patch.dict(os.environ, {}, clear=True):
                segments, _ = api_server._run_transcription_sync(
                    "meeting.mp3",
                    None,
                    None,
                    0.0,
                    "transcribe",
                    False,
                )
        finally:
            api_server._model = original_model

        self.assertEqual([segment.text for segment in segments], ["Hello."])
        self.assertFalse(calls[0]["condition_on_previous_text"])
        self.assertEqual(calls[0]["vad_parameters"]["min_silence_duration_ms"], 500)

    def test_transcription_env_can_restore_previous_text_conditioning(self):
        calls = []

        class Model:
            def transcribe(self, *args, **kwargs):
                calls.append(kwargs)
                return iter([]), SimpleNamespace(language="en")

        original_model = api_server._model
        api_server._model = Model()
        try:
            with patch.dict(
                os.environ,
                {
                    "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "true",
                    "WHISPER_VAD_MIN_SILENCE_MS": "2000",
                },
                clear=True,
            ):
                api_server._run_transcription_sync(
                    "meeting.mp3",
                    None,
                    None,
                    0.0,
                    "transcribe",
                    False,
                )
        finally:
            api_server._model = original_model

        self.assertTrue(calls[0]["condition_on_previous_text"])
        self.assertEqual(calls[0]["vad_parameters"]["min_silence_duration_ms"], 2000)


if __name__ == "__main__":
    unittest.main()
