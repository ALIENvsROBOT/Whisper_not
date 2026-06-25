# Ubuntu Production with sudo Podman

## Host prerequisites

Install:

- a supported NVIDIA driver;
- Podman;
- NVIDIA Container Toolkit, including `nvidia-ctk`.

Generate the rootful CDI specification:

```bash
sudo mkdir -p /etc/cdi
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
sudo nvidia-ctk cdi list
```

Verify GPU injection:

```bash
sudo podman run --rm \
  --security-opt=label=disable \
  --device=nvidia.com/gpu=all \
  docker.io/nvidia/cuda:12.9.1-base-ubuntu24.04 \
  nvidia-smi
```

Regenerate `/etc/cdi/nvidia.yaml` after NVIDIA driver changes.

## Pull the latest CUDA image

Public package:

```bash
sudo podman pull ghcr.io/alienvsrobot/whisper_not:cuda
```

If the GHCR package is private:

```bash
echo "$GITHUB_TOKEN" | sudo podman login ghcr.io \
  --username ALIENvsROBOT \
  --password-stdin

sudo podman pull ghcr.io/alienvsrobot/whisper_not:cuda
```

Use a classic GitHub personal access token with at least `read:packages`, or a
fine-grained token that can read the package.

## Accept and authorize the model

Open:

```text
large-v3-turbo
```

Accept the repository conditions and create a Hugging Face read token.

Create a root-only environment file:

```bash
sudo install -m 600 /dev/null /etc/whisper-not.env
sudoedit /etc/whisper-not.env
```

Contents:

```env
WHISPER_API_KEY=replace-with-a-long-random-secret
HF_TOKEN=hf_replace_with_your_read_token
WHISPER_MODEL=large-v3-turbo
WHISPER_LANGUAGE=auto
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_THREADS=4
WHISPER_BEAM=5
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
WHISPER_VAD_MIN_SILENCE_MS=500
WHISPER_DIARIZATION=on_demand
WHISPER_DIARIZATION_DEVICE=auto
WHISPER_DIARIZATION_THREADS=4
WHISPER_DIARIZATION_TIMEOUT_SECONDS=7200
WHISPER_WORD_TIMESTAMPS=false
WHISPER_MAX_UPLOAD_MB=1024
WHISPER_MAX_QUEUED_REQUESTS=15
WHISPER_QUEUE_TIMEOUT_SECONDS=7200
WHISPER_STARTUP_TIMEOUT=3600
WHISPER_LOG_LEVEL=INFO
```

## Create persistent storage

```bash
sudo podman volume create whisper-not-data
```

## Run the service

```bash
sudo podman run -d \
  --name whisper-not \
  --replace \
  --restart=unless-stopped \
  --security-opt=label=disable \
  --device=nvidia.com/gpu=all \
  --env-file=/etc/whisper-not.env \
  -p 9000:9000 \
  -v whisper-not-data:/var/lib/whisper \
  --tmpfs /run/whisper-temp:rw,size=2g,mode=1777 \
  ghcr.io/alienvsrobot/whisper_not:cuda
```

To override only the model without editing the environment file:

```bash
sudo podman run ... \
  -e WHISPER_MODEL="large-v3-turbo" \
  ghcr.io/alienvsrobot/whisper_not:cuda
```

The first start downloads the gated Whisper model and may take several
minutes.

Watch startup:

```bash
sudo podman logs -f whisper-not
```

Check health:

```bash
curl http://127.0.0.1:9000/health
```

Verify that the container sees the GPU:

```bash
sudo podman exec whisper-not nvidia-smi
```

## Test transcription

```bash
set -a
. /etc/whisper-not.env
set +a

curl http://127.0.0.1:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F file=@sample.wav \
  -F model=whisper-1
```

## Open WebUI

Use:

```text
API Base URL: http://UBUNTU_SERVER_IP:9000/v1
API Key: value of WHISPER_API_KEY
STT Model: whisper-1
```

The default Open WebUI request does not run timestamps or diarization.

## Automatic restart after reboot

Enable Podman’s restart service:

```bash
sudo systemctl enable --now podman-restart.service
```

The container’s `--restart=unless-stopped` policy then restores it after
reboot.

## Update to the latest image

```bash
sudo podman pull ghcr.io/alienvsrobot/whisper_not:cuda

sudo podman stop whisper-not

sudo podman run -d \
  --name whisper-not \
  --replace \
  --restart=unless-stopped \
  --security-opt=label=disable \
  --device=nvidia.com/gpu=all \
  --env-file=/etc/whisper-not.env \
  -p 9000:9000 \
  -v whisper-not-data:/var/lib/whisper \
  --tmpfs /run/whisper-temp:rw,size=2g,mode=1777 \
  ghcr.io/alienvsrobot/whisper_not:cuda
```

The named volume preserves model downloads and diarization models.

For deterministic production deployments, replace `:cuda` with an immutable
tag such as `:cuda-1.0.0`.
