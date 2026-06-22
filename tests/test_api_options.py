import unittest

from fastapi import HTTPException

import api_server


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


if __name__ == "__main__":
    unittest.main()
