# Implementation Architecture

## Objectives

The service has two distinct paths:

1. A low-latency Open WebUI path returning only `{"text":"..."}`.
2. An explicit rich-output path for timestamps, diarization, subtitles, and
   downloadable API responses.

Optional processing is request-scoped so the common path remains equivalent
to a direct faster-whisper transcription.

## Components

### `src/whisper_not/api.py`

Owns:

- FastAPI routes and bearer authentication;
- multipart request validation;
- request-option precedence;
- temporary upload lifecycle;
- faster-whisper inference;
- streaming SSE responses;
- response serialization and download headers;
- diarization orchestration.

The Whisper model is created once during application startup.

### `src/whisper_not/diarization.py`

Owns:

- downloading the sherpa-onnx segmentation and embedding models;
- decoding and resampling audio to mono 16 kHz;
- caching diarization pipelines by speaker count and threshold;
- speaker turn inference;
- assigning speaker labels to transcription segments by time overlap.

It uses ONNX Runtime and does not require PyTorch or a pyannote Hugging Face
token.

The CUDA image installs sherpa-onnx’s CUDA 12/cuDNN 9 wheel. The segmentation
and speaker-embedding model configurations both receive `provider="cuda"`
when CUDA is selected. Clustering and orchestration can still consume CPU.

### `scripts/container-entrypoint.sh`

Owns:

- container-only startup validation;
- environment normalization;
- CPU/CUDA compute defaults;
- API-key persistence;
- model cache configuration;
- startup readiness polling;
- graceful shutdown.

It accepts built-in faster-whisper names, Hugging Face CTranslate2 repository
IDs, and container-local model paths.

### `scripts/whisper-manage.sh`

Provides operational commands for:

- displaying server and API-key information;
- listing known models;
- downloading a model into the persistent cache;
- pre-downloading diarization models.

## Request flow

```text
Client
  |
  v
FastAPI multipart validation
  |
  v
RequestOptions resolution
  |
  +--> default: word timestamps off, diarization off
  |
  v
Upload copied to temporary filesystem
  |
  v
faster-whisper transcription under inference lock
  |
  +--> optional word timestamp generation
  |
  +--> optional sherpa-onnx diarization in isolated worker process
  |
  v
Response serializer
  |
  +--> JSON / text / SRT / VTT / diarized JSON
  |
  v
Temporary upload removed
```

## Option precedence

Word timestamps:

1. `timestamp_granularities[]=word`;
2. global `WHISPER_WORD_TIMESTAMPS=true`;
3. disabled.

Diarization:

1. `WHISPER_DIARIZATION=disabled` rejects all diarization requests.
2. `WHISPER_DIARIZATION=always` diarizes every non-streaming request.
3. In `on_demand` mode, `diarize=true`, `diarized_json`, or
   `gpt-4o-transcribe-diarize` activates diarization.
4. Otherwise diarization remains off.

Speaker clustering is not capped at two speakers. A positive
`num_speakers` sets an exact cluster count; `-1` enables threshold-based
automatic detection. CUDA diarization is isolated from CTranslate2 because
initializing both native CUDA runtimes in one process can terminate that
process after an earlier transcription.

Language:

1. request `language`;
2. `WHISPER_LANGUAGE`;
3. faster-whisper automatic detection.

## Concurrency

CTranslate2 inference is serialized through `_inference_lock`. This avoids
unsafe concurrent calls and large, unpredictable GPU memory spikes.

Diarization runs in one lazy, persistent spawned process. This isolates
sherpa-onnx/ONNX Runtime CUDA state from CTranslate2 CUDA state and prevents a
native diarization failure from terminating the API/model process. The worker
caches its model pipelines until container shutdown and restarts after an
unexpected exit.

Blocking Whisper calls and diarization IPC execute through `asyncio.to_thread()`.
The event loop therefore remains available for health checks, authentication,
request validation, and other lightweight work during long inference jobs.

An ASGI admission middleware runs before multipart parsing. One audio request
is active at a time, and `WHISPER_MAX_QUEUED_REQUESTS` bounds callers waiting
behind it. Requests beyond that bound receive HTTP 429 before their upload is
read. `WHISPER_QUEUE_TIMEOUT_SECONDS` prevents abandoned callers from waiting
indefinitely. The admission slot is held until normal or streaming responses
finish, so transcription remains strictly serial.

For higher throughput, deploy multiple replicas and route requests across
them. Each replica needs enough GPU memory for its own Whisper model.

## Temporary and persistent data

Persistent:

```text
/var/lib/whisper
```

Contains:

- Hugging Face model cache;
- diarization ONNX models;
- generated API-key state;
- operational metadata.

Temporary:

```text
/run/whisper-temp
```

Contains uploaded audio only while a request is active. Compose mounts it as
`tmpfs`; request cleanup removes files after transcription and diarization.

Generated transcript files are not stored.

## GPU execution

The CUDA image contains CUDA 12 and cuDNN runtime libraries required by
CTranslate2. Recommended settings:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

GPU access can be supplied through:

- CDI: `nvidia.com/gpu=all`;
- Docker GPU reservation / `--gpus all`.

The application itself uses the same code path on Windows Docker Desktop and
Ubuntu; only host GPU injection differs.

## Model loading

`WHISPER_MODEL` is passed directly to `faster_whisper.WhisperModel`.

Supported forms:

```text
large-v3-turbo
TheChola/whisper-large-v3-turbo-german-faster-whisper
/models/custom-ctranslate2
```

`HF_TOKEN` is inherited by Hugging Face Hub for gated or private models.
`WHISPER_LOCAL_ONLY` prevents network downloads and requires a populated
cache.

## Testing strategy

Unit tests verify:

- default optional features remain disabled;
- request option validation and precedence;
- OpenAI diarization aliases;
- response payloads and attachment filenames.

FastAPI integration tests use a deterministic fake Whisper model to verify:

- Open WebUI request compatibility;
- `/v1/models`;
- word timestamp activation;
- diarized JSON responses.

Container verification uses actual faster-whisper inference with a generated
speech WAV. CUDA verification additionally confirms NVIDIA visibility and
CTranslate2 GPU execution.

## Failure behavior

The server fails explicitly:

- malformed options return `400`;
- unauthorized requests return `401`;
- oversized uploads return `413`;
- unavailable startup model returns `503`;
- inference and diarization failures return `500` with logged context.

Temporary files are removed in `finally` blocks even when processing fails.
