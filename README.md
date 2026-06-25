# Whisper_not

GPU-first, self-hosted speech-to-text server powered by faster-whisper. It
provides an OpenAI-compatible endpoint for Open WebUI and optional
request-scoped word timestamps, speaker diarization, subtitle output, and
downloadable responses.

## Behavior

The default request is intentionally fast:

```http
POST /v1/audio/transcriptions
model=whisper-1
file=<audio>
```

Response:

```json
{"text":"Transcribed text"}
```

Word timestamps and diarization are disabled for this path. Open WebUI does
not pay their computation or latency cost.

Optional direct API requests can enable:

- faster-whisper word timestamps;
- local sherpa-onnx speaker diarization;
- OpenAI-style `diarized_json`;
- `json`, `text`, `verbose_json`, `srt`, and `vtt`;
- attachment responses with `download=true`.

## Recommended model

The default GPU stack uses:

```text
large-v3-turbo
```

It is the multilingual large-v3 turbo model supported directly by
faster-whisper. It is the best default when the service may receive German,
English, or other languages.

Important:

- Use `WHISPER_LANGUAGE=auto` for mixed-language uploads.
- Set `WHISPER_LANGUAGE=de` only if all production audio is German and you want
  to skip language detection.
- A Hugging Face token is only needed for private or gated Hugging Face models.

Any compatible faster-whisper built-in model, Hugging Face CTranslate2
repository ID, or mounted local model path can be supplied through
`WHISPER_MODEL`.

Set it through Compose/Portainer, `--env-file .env`, or directly:

```bash
-e WHISPER_MODEL="large-v3-turbo"
```

## Portainer GPU stack

Use [deploy/compose.cuda-cdi.yml](deploy/compose.cuda-cdi.yml) on a
CDI-configured Linux host. On Windows
Docker Desktop, use
[deploy/compose.cuda.yml](deploy/compose.cuda.yml), which was verified
with this project’s local CUDA test.

Set these Portainer environment variables:

| Variable | Required | Recommended value |
|---|---:|---|
| `WHISPER_API_KEY` | Yes | A long random secret |
| `HF_TOKEN` | For gated models | Hugging Face read token |
| `WHISPER_MODEL` | No | `large-v3-turbo` |
| `WHISPER_HOST_PORT` | No | `9000` |
| `WHISPER_THREADS` | No | `4` |
| `WHISPER_COMPUTE_TYPE` | No | `float16` |

The primary stack requests all GPUs through CDI:

```yaml
devices:
  - nvidia.com/gpu=all
```

This matches:

```powershell
docker run --device nvidia.com/gpu=all ...
```

If an Ubuntu Docker installation does not expose CDI devices, use
[deploy/compose.cuda.yml](deploy/compose.cuda.yml), which uses the standard
NVIDIA Compose reservation.

For a rootful Ubuntu Podman deployment using
`sudo podman --device=nvidia.com/gpu=all`, follow
[docs/PODMAN_PRODUCTION.md](docs/PODMAN_PRODUCTION.md).

## Windows Docker Desktop prerequisites

1. Use Windows 11 or a current supported Windows 10 release.
2. Enable Docker Desktop’s WSL 2 backend.
3. Install a current NVIDIA Windows driver with WSL CUDA support.
4. Update WSL:

   ```powershell
   wsl --update
   ```

5. Verify GPU access:

   ```powershell
   docker run --rm --device nvidia.com/gpu=all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
   ```

## Ubuntu prerequisites

Install the NVIDIA driver and NVIDIA Container Toolkit, configure Docker, and
verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

For CDI, generate and inspect the NVIDIA CDI specification:

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list
```

## Open WebUI v0.9.6

In **Admin Panel → Settings → Audio → Speech-to-Text**:

| Setting | Value |
|---|---|
| Engine | `OpenAI` |
| API Base URL | `http://HOST_OR_IP:9000/v1` |
| API Key | Value of `WHISPER_API_KEY` |
| STT Model | `whisper-1` |

If both containers share a Docker network, use:

```text
http://whisper-not:9000/v1
```

Equivalent Open WebUI environment variables:

```env
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://whisper-not:9000/v1
AUDIO_STT_OPENAI_API_KEY=your-secret
AUDIO_STT_MODEL=whisper-1
```

Open WebUI sends only the model, optional language, and file. The server
returns the expected `{"text":"..."}` JSON response.

## Direct API examples

Set:

```bash
BASE_URL=http://localhost:9000/v1
API_KEY=your-secret
```

Plain transcript:

```bash
curl "$BASE_URL/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@meeting.mp3 \
  -F model=whisper-1
```

Word timestamps:

```bash
curl "$BASE_URL/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@meeting.mp3 \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"
```

Speaker diarization:

```bash
curl "$BASE_URL/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@meeting.mp3 \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F diarize=true
```

OpenAI-style diarized JSON:

```bash
curl "$BASE_URL/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@meeting.mp3 \
  -F model=gpt-4o-transcribe-diarize \
  -F response_format=diarized_json \
  -F num_speakers=4
```

Omit `num_speakers` for automatic multi-speaker detection. Diarization is not
limited to two participants.

Download SRT:

```bash
curl "$BASE_URL/audio/transcriptions" \
  -H "Authorization: Bearer $API_KEY" \
  -F file=@meeting.mp3 \
  -F model=whisper-1 \
  -F response_format=srt \
  -F diarize=true \
  -F download=true \
  --remote-header-name \
  --remote-name
```

No generated output is retained by the server. Uploaded audio is stored only
in the container’s temporary filesystem while the request is processed.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | Built-in name, Hugging Face ID, or local model path |
| `WHISPER_LANGUAGE` | `auto` | Auto-detect or ISO-639-1 language code |
| `WHISPER_DEVICE` | `cpu` | `cpu`, `cuda`, or `auto` |
| `WHISPER_COMPUTE_TYPE` | CPU `int8`, CUDA `float16` | CTranslate2 compute type |
| `WHISPER_BEAM` | `5` | Decoding beam size |
| `WHISPER_CONDITION_ON_PREVIOUS_TEXT` | `false` | Disable prior-window conditioning to reduce long-audio repetition loops |
| `WHISPER_VAD_MIN_SILENCE_MS` | `500` | VAD silence threshold passed to faster-whisper |
| `WHISPER_THREADS` | `2` | CPU inference threads |
| `WHISPER_DIARIZATION` | `on_demand` | `on_demand`, `always`, or `disabled` |
| `WHISPER_DIARIZATION_DEVICE` | `auto` | `auto`, `cuda`, or `cpu`; auto follows the Whisper device |
| `WHISPER_DIARIZATION_THREADS` | Same as `WHISPER_THREADS` | ONNX Runtime threads for diarization |
| `WHISPER_DIARIZATION_TIMEOUT_SECONDS` | `7200` | Maximum duration of one isolated diarization operation |
| `WHISPER_WORD_TIMESTAMPS` | `false` | Global word timestamps; per-request is recommended |
| `WHISPER_API_KEY` | Generated for new persistent volumes | Bearer-token authentication |
| `HF_TOKEN` | Empty | Hugging Face token for gated/private models |
| `WHISPER_MAX_UPLOAD_MB` | `1024` | Maximum uploaded audio size; `0` disables the limit |
| `WHISPER_MAX_QUEUED_REQUESTS` | `15` | Requests allowed to wait behind the single active transcription |
| `WHISPER_QUEUE_TIMEOUT_SECONDS` | `7200` | Maximum time a queued request may wait |
| `WHISPER_STARTUP_TIMEOUT` | `3600` | Seconds allowed for initial model download and loading |

## Container images and versions

GitHub Actions publishes:

```text
ghcr.io/alienvsrobot/whisper_not:latest
ghcr.io/alienvsrobot/whisper_not:cuda
```

Every successful `main` workflow also publishes automatic immutable versions:

```text
0.1.<workflow-run-number>
cuda-0.1.<workflow-run-number>
sha-<commit>
cuda-sha-<commit>
```

Git tags such as `v1.2.3` additionally publish `1.2.3`, `1.2`, `cuda-1.2.3`,
and `cuda-1.2`.

## Production notes

- Put the API behind HTTPS when exposed outside a trusted network.
- Keep `WHISPER_API_KEY` and `HF_TOKEN` out of Compose files and Git.
- Persist `/var/lib/whisper` to avoid downloading models after restarts.
- Keep diarization on demand for the fastest Open WebUI path.
- Set an exact `num_speakers` when known for more reliable clustering.
- A single process intentionally serializes CTranslate2 inference to avoid
  unsafe concurrent access and unpredictable GPU memory spikes.
- Blocking transcription runs in a worker thread and diarization runs in an
  isolated process, so health checks and lightweight API handling remain
  responsive during long recordings.
- Admission control accepts one active audio request and up to 15 waiting
  requests by default. Additional requests receive HTTP 429 before upload
  bodies are consumed.

The CUDA image installs the official sherpa-onnx CUDA 12/cuDNN 9 wheel.
Diarization segmentation and speaker-embedding inference use CUDA when
`WHISPER_DIARIZATION_DEVICE=auto` and `WHISPER_DEVICE=cuda`. Clustering and
pipeline coordination still perform some CPU work.

## Documentation

- [API usage](docs/API_USAGE.md)
- [CI/CD instructions](docs/CI_INSTRUCTIONS.md)
- [Implementation architecture](docs/IMPLEMENTATION_ARCHITECTURE.md)
- [Pull, run, and change models](docs/PULL_AND_USE.md)
- [Open WebUI integration](docs/OPEN_WEBUI_INTEGRATION.md)
- [Ubuntu production with sudo Podman](docs/PODMAN_PRODUCTION.md)
