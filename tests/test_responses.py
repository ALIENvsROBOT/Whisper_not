import json
import unittest
from types import SimpleNamespace

import api_server


def _segment(index, start, end, text):
    return SimpleNamespace(
        id=index,
        seek=0,
        start=start,
        end=end,
        text=text,
        tokens=[1, 2],
        temperature=0.0,
        avg_logprob=-0.1,
        compression_ratio=1.0,
        no_speech_prob=0.01,
        words=None,
    )


class ResponseTests(unittest.TestCase):
    def test_attachment_header_uses_safe_stem_and_extension(self):
        headers = api_server._download_headers("../../meeting final.mp3", "srt")

        self.assertEqual(
            headers["Content-Disposition"],
            'attachment; filename="meeting_final.srt"',
        )

    def test_no_attachment_header_without_download(self):
        self.assertEqual(api_server._download_headers("meeting.mp3", None), {})

    def test_diarized_json_matches_openai_segment_shape(self):
        segments = [
            _segment(0, 0.0, 1.25, "Hello."),
            _segment(1, 1.25, 2.5, "Hi."),
        ]
        payload = api_server._build_diarized_payload(
            segments=segments,
            speaker_map={0: "SPEAKER_00", 1: "SPEAKER_01"},
            duration=2.5,
        )

        self.assertEqual(payload["task"], "transcribe")
        self.assertEqual(payload["duration"], 2.5)
        self.assertEqual(payload["segments"][0]["type"], "transcript.text.segment")
        self.assertEqual(payload["segments"][0]["speaker"], "A")
        self.assertEqual(payload["segments"][1]["speaker"], "B")
        self.assertEqual(payload["text"], "A: Hello.\nB: Hi.")

    def test_json_download_serializes_as_json_attachment(self):
        response = api_server._json_response(
            {"text": "hello"},
            original_name="sample.wav",
            download=True,
            response_format="json",
        )

        self.assertEqual(response.media_type, "application/json")
        self.assertIn("sample.json", response.headers["content-disposition"])
        self.assertEqual(json.loads(response.body), {"text": "hello"})


if __name__ == "__main__":
    unittest.main()
