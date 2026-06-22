# CI/CD Instructions

## Overview

The repository contains two GitHub Actions workflows:

- `.github/workflows/ci.yml` validates source, API behavior, workflow syntax,
  Compose files, shell scripts, and both container images.
- `.github/workflows/container.yml` repeats the required validation gate,
  then builds and publishes CPU and CUDA images to GitHub Container Registry.

## Required repository settings

In GitHub:

1. Open **Settings → Actions → General**.
2. Under **Workflow permissions**, select **Read and write permissions**.
3. Keep **Allow GitHub Actions to create and approve pull requests** disabled;
   these workflows do not require it.
4. Retain the default untrusted pull-request security policy unless the
   repository has a specific reason to change it.

No registry password is required. Publishing uses the repository-scoped
`GITHUB_TOKEN` with `packages: write`.

## CI workflow

Triggers:

- every pull request;
- every push to `main`.

The `test` job:

1. Uses Python 3.12.
2. Installs pinned runtime dependencies plus `httpx`.
3. Runs all unit and FastAPI compatibility tests.
4. Compiles `src/whisper_not`.
5. Runs ShellCheck against scripts under `scripts/`.
6. Validates both Compose files.
7. Validates both workflow files with `actionlint`.

The `docker` job:

1. Configures Docker Buildx.
2. Builds the CPU image.
3. Builds the CUDA image without running it.

CUDA runtime execution is not attempted on GitHub-hosted runners because they
do not provide an NVIDIA GPU.

## Container publishing workflow

The publishing workflow first runs tests, source compilation, ShellCheck,
Compose validation, and `actionlint`. CPU and CUDA publication jobs depend on
that validation job, so no image is published when validation fails.

Triggers:

- each push to `main`;
- semantic tags matching `v*.*.*`;
- manual **Run workflow** requests.

Published registry:

```text
ghcr.io/alienvsrobot/whisper_not
```

CPU tags on `main`:

```text
latest
0.1.<workflow-run-number>
sha-<short-commit>
```

CUDA tags on `main`:

```text
cuda
cuda-0.1.<workflow-run-number>
cuda-sha-<short-commit>
```

For `v1.2.3`, the workflow also publishes:

```text
1.2.3
1.2
cuda-1.2.3
cuda-1.2
```

CPU images are multi-platform:

```text
linux/amd64
linux/arm64
```

CUDA images are:

```text
linux/amd64
```

## First publication

After the first successful publishing workflow:

1. Open the package under the GitHub account’s **Packages** section.
2. Open **Package settings**.
3. Change visibility to **Public** if anonymous Portainer pulls are required.
4. Connect the package to the repository if GitHub has not done so
   automatically.

## Creating a stable release

Run:

```bash
git tag -a v1.0.0 -m "v1.0.0"
git push origin v1.0.0
```

The tag triggers immutable CPU and CUDA version tags. Deploy production stacks
with a full immutable tag:

```text
ghcr.io/alienvsrobot/whisper_not:cuda-1.0.0
```

Avoid `latest` or `cuda` when deterministic rollback is required.

## Rollback

Change `WHISPER_IMAGE` in Portainer to an earlier immutable tag and redeploy:

```text
WHISPER_IMAGE=ghcr.io/alienvsrobot/whisper_not:cuda-0.1.42
```

The model cache volume remains compatible because model artifacts are stored
separately under `/var/lib/whisper`.

## Troubleshooting

### Package push receives HTTP 403

- Confirm workflow permissions allow writes.
- Confirm the workflow has `packages: write`.
- Confirm the package is linked to the repository.

### CUDA build succeeds but runtime has no GPU

Container building does not prove host GPU availability. Validate the target:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

For CDI:

```bash
nvidia-ctk cdi list
docker run --rm --device nvidia.com/gpu=all \
  nvidia/cuda:12.9.1-base-ubuntu24.04 nvidia-smi
```

### Dependency update

Update one pinned package at a time under `requirements/`, then run:

```bash
python -m unittest discover -s tests -v
docker build -t whisper-not:test -f docker/Dockerfile.cpu .
docker build -t whisper-not:cuda-test -f docker/Dockerfile.cuda .
```

Do not merge dependency updates based only on successful installation; retain
the API and live transcription verification.
