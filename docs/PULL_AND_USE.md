# Pull and Use

## Image names

CPU:

```text
ghcr.io/alienvsrobot/whisper_not:latest
```

NVIDIA CUDA:

```text
ghcr.io/alienvsrobot/whisper_not:cuda
```

For production, prefer an immutable version such as:

```text
ghcr.io/alienvsrobot/whisper_not:cuda-1.0.0
```

## Docker Desktop on Windows

Pull:

```powershell
docker pull ghcr.io/alienvsrobot/whisper_not:cuda
```

Create a persistent volume:

```powershell
docker volume create whisper-not-data
```

Run:

```powershell
docker run -d `
  --name whisper-not `
  --restart unless-stopped `
  --gpus all `
  -p 9000:9000 `
  -v whisper-not-data:/var/lib/whisper `
  -e WHISPER_API_KEY="replace-with-a-long-secret" `
  -e HF_TOKEN="hf_replace_with_your_token" `
  -e WHISPER_MODEL="TheChola/whisper-large-v3-turbo-german-faster-whisper" `
  -e WHISPER_LANGUAGE="auto" `
  -e WHISPER_DEVICE="cuda" `
  -e WHISPER_COMPUTE_TYPE="float16" `
  -e WHISPER_DIARIZATION="on_demand" `
  ghcr.io/alienvsrobot/whisper_not:cuda
```

Docker Desktop on this project’s verified Windows environment exposes NVIDIA
GPUs with `--gpus all`. CDI syntax requires a resolvable host CDI
specification and is not present in every Docker Desktop installation.

## Ubuntu with sudo Podman

Pull:

```bash
sudo podman pull ghcr.io/alienvsrobot/whisper_not:cuda
```

Run:

```bash
sudo podman run -d \
  --name whisper-not \
  --replace \
  --restart=unless-stopped \
  --security-opt=label=disable \
  --device=nvidia.com/gpu=all \
  -p 9000:9000 \
  -v whisper-not-data:/var/lib/whisper \
  --env-file=/etc/whisper-not.env \
  ghcr.io/alienvsrobot/whisper_not:cuda
```

See [PODMAN_PRODUCTION.md](PODMAN_PRODUCTION.md) for full host setup.

## Change the model

The model is selected once when the container starts:

```env
WHISPER_MODEL=large-v3-turbo
```

Docker or Podman environment option:

```bash
-e WHISPER_MODEL="large-v3-turbo"
```

Hugging Face CTranslate2 model:

```env
WHISPER_MODEL=TheChola/whisper-large-v3-turbo-german-faster-whisper
HF_TOKEN=hf_replace_with_your_token
WHISPER_STARTUP_TIMEOUT=3600
```

Mounted local model:

```bash
-v /srv/whisper-models:/models:ro \
-e WHISPER_MODEL=/models/my-ctranslate2-model
```

Restart or replace the container after changing `WHISPER_MODEL`.

The API `model` field is a compatibility field. It does not dynamically swap
the startup model. This avoids repeated model loading and unstable GPU memory
usage.

## Use an environment file

Copy `deploy/env.example` to `.env`, set the API key, token, and model, then use:

```bash
docker run --env-file .env ...
```

or:

```bash
sudo podman run --env-file .env ...
```

Docker Compose automatically reads `.env` for `${VARIABLE}` substitution:

```bash
docker compose -f deploy/compose.cuda.yml up -d
```

The container startup script also supports a bind-mounted shell environment
file:

```bash
-v /absolute/path/whisper.env:/whisper.env:ro
```

Direct `-e WHISPER_MODEL="..."` values override image defaults. For a
bind-mounted `/whisper.env`, values in that file are sourced by the startup
script and take precedence.

## Verify

Health:

```bash
curl http://127.0.0.1:9000/health
```

Models:

```bash
curl http://127.0.0.1:9000/v1/models \
  -H "Authorization: Bearer $WHISPER_API_KEY"
```

Transcription:

```bash
curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F file=@sample.wav \
  -F model=whisper-1
```

Word timestamps:

```bash
curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F file=@sample.wav \
  -F model=whisper-1 \
  -F response_format=verbose_json \
  -F "timestamp_granularities[]=word"
```

Diarization:

```bash
curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F file=@meeting.wav \
  -F model=gpt-4o-transcribe-diarize \
  -F response_format=diarized_json
```

See [API_USAGE.md](API_USAGE.md) for all parameters and response
formats.
