import unittest
from unittest.mock import MagicMock, patch
import sys

sys.modules.setdefault("sherpa_onnx", MagicMock())
from whisper_not import diarization as diarizer


class DiarizationProviderTests(unittest.TestCase):
    def setUp(self):
        self.original_sherpa = diarizer.sherpa_onnx
        diarizer.sherpa_onnx = sys.modules["sherpa_onnx"]

    def tearDown(self):
        diarizer.sherpa_onnx = self.original_sherpa
        diarizer._pipelines.clear()

    def test_auto_uses_cuda_when_whisper_uses_cuda(self):
        self.assertEqual(
            diarizer.resolve_provider("auto", whisper_device="cuda"),
            "cuda",
        )

    def test_auto_uses_cpu_when_whisper_uses_cpu(self):
        self.assertEqual(
            diarizer.resolve_provider("auto", whisper_device="cpu"),
            "cpu",
        )

    def test_invalid_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "WHISPER_DIARIZATION_DEVICE"):
            diarizer.resolve_provider("directml", whisper_device="cuda")

    def test_provider_is_passed_to_all_model_configurations(self):
        fake_config = MagicMock()
        fake_config.validate.return_value = True
        fake_pipeline = MagicMock()
        fake_pipeline.sample_rate = 16000
        with (
            patch.object(diarizer, "_ensure_models", return_value=("seg.onnx", "emb.onnx")),
            patch.object(diarizer, "_cuda_provider_available", return_value=True),
            patch.object(
                diarizer.sherpa_onnx,
                "OfflineSpeakerSegmentationModelConfig",
            ) as segmentation_config,
            patch.object(
                diarizer.sherpa_onnx,
                "OfflineSpeakerSegmentationPyannoteModelConfig",
            ),
            patch.object(
                diarizer.sherpa_onnx,
                "SpeakerEmbeddingExtractorConfig",
            ) as embedding_config,
            patch.object(
                diarizer.sherpa_onnx,
                "FastClusteringConfig",
            ) as clustering_config,
            patch.object(
                diarizer.sherpa_onnx,
                "OfflineSpeakerDiarizationConfig",
                return_value=fake_config,
            ),
            patch.object(
                diarizer.sherpa_onnx,
                "OfflineSpeakerDiarization",
                return_value=fake_pipeline,
            ),
        ):
            diarizer._pipelines.clear()
            diarizer.load(cache_dir=".", provider="cuda", num_speakers=4)

        self.assertEqual(segmentation_config.call_args.kwargs["provider"], "cuda")
        self.assertEqual(embedding_config.call_args.kwargs["provider"], "cuda")
        self.assertEqual(clustering_config.call_args.kwargs["num_clusters"], 4)

    def test_cuda_provider_fails_loudly_when_unavailable(self):
        with patch.object(diarizer, "_cuda_provider_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA execution is not available"):
                diarizer.load(cache_dir=".", provider="cuda")

    def test_cuda_wheel_requires_visible_gpu(self):
        class FakeRuntime:
            __version__ = "1.13.3+cuda12.cudnn9"

        with patch.object(diarizer, "_nvidia_smi_available", return_value=True):
            self.assertTrue(diarizer._cuda_provider_available(FakeRuntime()))

        with patch.object(diarizer, "_nvidia_smi_available", return_value=False):
            self.assertFalse(diarizer._cuda_provider_available(FakeRuntime()))

    def test_cpu_wheel_does_not_claim_cuda(self):
        class FakeRuntime:
            __version__ = "1.13.3"

            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

        self.assertFalse(diarizer._cuda_provider_available(FakeRuntime()))

    def test_assign_speakers_preserves_more_than_two_speaker_labels(self):
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
            {"start": 2.0, "end": 3.0},
            {"start": 3.0, "end": 4.0},
        ]
        turns = [
            (0.0, 1.0, "SPEAKER_00"),
            (1.0, 2.0, "SPEAKER_01"),
            (2.0, 3.0, "SPEAKER_02"),
            (3.0, 4.0, "SPEAKER_03"),
        ]

        diarizer.assign_speakers(segments, turns)

        self.assertEqual(
            [segment["speaker"] for segment in segments],
            ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"],
        )


if __name__ == "__main__":
    unittest.main()
