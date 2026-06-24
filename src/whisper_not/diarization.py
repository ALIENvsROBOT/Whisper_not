"""
Speaker diarization via sherpa-onnx (ONNX Runtime, no PyTorch).

Downloads ONNX models on first use and provides speaker-segment alignment
for Whisper transcription output.

Project: https://github.com/ALIENvsROBOT/Whisper_not
"""

import logging
import os
import subprocess
import tarfile
import time

import numpy as np

sherpa_onnx = None

logger = logging.getLogger("whisper_server.diarizer")

# ---------------------------------------------------------------------------
# Model URLs (GitHub releases from k2-fsa/sherpa-onnx)
# ---------------------------------------------------------------------------

_SEG_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_EMB_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)

_SEG_MODEL_REL = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
_EMB_MODEL_REL = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_pipelines = {}


def _get_sherpa_onnx():
    global sherpa_onnx
    if sherpa_onnx is None:
        try:
            import sherpa_onnx as runtime
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "sherpa-onnx is not installed in this runtime."
            ) from exc
        sherpa_onnx = runtime
    return sherpa_onnx


def resolve_provider(value: str = "auto", whisper_device: str = "cpu") -> str:
    """Resolve the sherpa-onnx execution provider."""
    requested = (value or "auto").strip().lower()
    if requested == "auto":
        return "cuda" if whisper_device.strip().lower() == "cuda" else "cpu"
    if requested not in {"cpu", "cuda"}:
        raise ValueError(
            "WHISPER_DIARIZATION_DEVICE must be one of: auto, cpu, cuda."
        )
    return requested


def _nvidia_smi_available() -> bool:
    try:
        subprocess.run(
            ["nvidia-smi"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


def _cuda_provider_available(runtime) -> bool:
    """Return True when the sherpa runtime can execute diarization on CUDA."""
    provider_names = None
    if callable(getattr(runtime, "get_available_providers", None)):
        provider_names = runtime.get_available_providers()

    if provider_names is not None:
        providers = {str(provider).lower() for provider in provider_names}
        if "cudaexecutionprovider" in providers or "cuda" in providers:
            return True

    # The CUDA sherpa-onnx wheel bundles its own CUDA runtime path; the separate
    # onnxruntime package installed for faster-whisper may still report CPU-only.
    runtime_version = str(getattr(runtime, "__version__", "")).lower()
    if "+cuda" in runtime_version:
        return _nvidia_smi_available()

    if provider_names is None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            return False
        else:
            provider_names = ort.get_available_providers()
            return "cudaexecutionprovider" in {
                str(provider).lower() for provider in provider_names
            }

    return False


def _download_file(url: str, dest: str) -> None:
    """Download a file using curl (available in the container)."""
    logger.info("Downloading %s -> %s", url, dest)
    subprocess.run(
        ["curl", "-fSL", "--retry", "3", "--retry-delay", "2", "-o", dest, url],
        check=True,
    )


def _ensure_models(cache_dir: str) -> tuple:
    """Download ONNX models if not already present. Returns (seg_path, emb_path)."""
    seg_path = os.path.join(cache_dir, _SEG_MODEL_REL)
    emb_path = os.path.join(cache_dir, _EMB_MODEL_REL)

    if not os.path.isfile(seg_path):
        archive = os.path.join(cache_dir, "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
        _download_file(_SEG_MODEL_URL, archive)
        logger.info("Extracting segmentation model...")
        with tarfile.open(archive, "r:bz2") as tar:
            tar.extractall(path=cache_dir, filter="data")
        os.unlink(archive)

    if not os.path.isfile(emb_path):
        _download_file(_EMB_MODEL_URL, emb_path)

    return seg_path, emb_path


def load(
    cache_dir: str = "/var/lib/whisper",
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
    provider: str = "cpu",
    num_threads: int = 2,
) -> object:
    """Return a cached diarization pipeline for the requested clustering options."""
    runtime = _get_sherpa_onnx()
    if provider == "cuda" and not _cuda_provider_available(runtime):
        raise RuntimeError(
            "WHISPER_DIARIZATION_DEVICE resolved to cuda, but CUDA execution "
            "is not available to sherpa-onnx in this container."
        )

    cache_dir = os.path.abspath(cache_dir)
    key = (cache_dir, num_speakers, cluster_threshold, provider, num_threads)
    if key in _pipelines:
        return _pipelines[key]

    seg_path, emb_path = _ensure_models(cache_dir)

    # If the speaker count is unknown, use threshold-based auto clustering.
    num_clusters = num_speakers if num_speakers > 0 else -1

    config = runtime.OfflineSpeakerDiarizationConfig(
        segmentation=runtime.OfflineSpeakerSegmentationModelConfig(
            pyannote=runtime.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=seg_path,
            ),
            provider=provider,
            num_threads=num_threads,
        ),
        embedding=runtime.SpeakerEmbeddingExtractorConfig(
            model=emb_path,
            provider=provider,
            num_threads=num_threads,
        ),
        clustering=runtime.FastClusteringConfig(
            num_clusters=num_clusters,
            threshold=cluster_threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )

    if not config.validate():
        raise RuntimeError(
            "Diarization config validation failed. Check that model files exist."
        )

    pipeline = runtime.OfflineSpeakerDiarization(config)
    _pipelines[key] = pipeline
    logger.info(
        "Diarization pipeline ready (provider=%s, sample_rate=%d, num_clusters=%d, threshold=%.2f)",
        provider,
        pipeline.sample_rate,
        num_clusters,
        cluster_threshold,
    )
    return pipeline


def is_loaded() -> bool:
    """Return True if at least one diarization pipeline is initialized."""
    return bool(_pipelines)


def _load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load and resample audio file to mono float32 at target_sr using PyAV.
    Uses the same approach as faster-whisper's decode_audio (AudioResampler).
    PyAV is already installed as a dependency of faster-whisper and supports
    all formats that FFmpeg supports (mp3, ogg, m4a, webm, wav, flac, etc.).
    """
    import io

    import av

    resampler = av.audio.resampler.AudioResampler(
        format="s16",
        layout="mono",
        rate=target_sr,
    )

    raw_buffer = io.BytesIO()
    with av.open(audio_path, mode="r", metadata_errors="ignore") as container:
        for frame in container.decode(audio=0):
            for resampled in resampler.resample(frame):
                raw_buffer.write(resampled.to_ndarray())
        # Flush remaining buffered samples from the resampler
        for resampled in resampler.resample(None):
            if resampled.samples > 0:
                raw_buffer.write(resampled.to_ndarray())

    audio = np.frombuffer(raw_buffer.getvalue(), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    return audio


def diarize(
    audio_path: str,
    cache_dir: str = "/var/lib/whisper",
    num_speakers: int = -1,
    cluster_threshold: float = 0.5,
    provider: str = "cpu",
    num_threads: int = 2,
):
    """
    Run diarization on an audio file.

    Returns a list of (start, end, speaker_label) tuples sorted by start time.
    """
    t0 = time.monotonic()
    pipeline = load(
        cache_dir=cache_dir,
        num_speakers=num_speakers,
        cluster_threshold=cluster_threshold,
        provider=provider,
        num_threads=num_threads,
    )
    t_loaded = time.monotonic()

    audio = _load_audio(audio_path, target_sr=pipeline.sample_rate)
    t_decoded = time.monotonic()

    result = pipeline.process(audio).sort_by_start_time()
    t_processed = time.monotonic()
    turns = []
    for r in result:
        turns.append((r.start, r.end, f"SPEAKER_{r.speaker:02d}"))
    logger.info(
        "Diarization timings | provider=%s threads=%d load=%.2fs decode=%.2fs process=%.2fs turns=%d total=%.2fs",
        provider,
        num_threads,
        t_loaded - t0,
        t_decoded - t_loaded,
        t_processed - t_decoded,
        len(turns),
        time.monotonic() - t0,
    )
    return turns


def assign_speakers(segments, diarization_turns):
    """
    Assign a speaker label to each Whisper segment based on maximum time overlap
    with diarization turns.

    Args:
        segments: list of segment dicts with 'start' and 'end' keys (seconds).
        diarization_turns: list of (start, end, speaker_label) tuples.

    Returns:
        The same segments list with a 'speaker' key added to each segment.
    """
    if not diarization_turns:
        for seg in segments:
            seg["speaker"] = "SPEAKER_00"
        return segments

    turn_index = 0
    turn_count = len(diarization_turns)

    for seg in segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0

        while turn_index < turn_count and diarization_turns[turn_index][1] <= seg_start:
            turn_index += 1

        scan_index = turn_index
        while scan_index < turn_count:
            turn_start, turn_end, speaker = diarization_turns[scan_index]
            if turn_start >= seg_end:
                break

            overlap_start = max(seg_start, turn_start)
            overlap_end = min(seg_end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
            scan_index += 1

        seg["speaker"] = best_speaker

    return segments
