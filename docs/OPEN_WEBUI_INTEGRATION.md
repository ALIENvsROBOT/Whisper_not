# Open WebUI Integration

## Supported behavior

Open WebUI v0.9.6 sends an OpenAI-compatible multipart request containing:

- the audio file;
- the configured STT model;
- an optional language.

Whisper_not returns:

```json
{"text":"Transcribed text"}
```

No diarization or word timestamp computation runs for this default request.

## Open WebUI settings

Open:

```text
Admin Panel → Settings → Audio → Speech-to-Text
```

Set:

| Setting | Value |
|---|---|
| Speech-to-Text Engine | `OpenAI` |
| API Base URL | `http://WHISPER_HOST:9000/v1` |
| API Key | The server’s `WHISPER_API_KEY` |
| STT Model | `whisper-1` |

Do not append `/audio/transcriptions` to the base URL. Open WebUI appends that
path itself.

## Environment configuration

```env
AUDIO_STT_ENGINE=openai
AUDIO_STT_OPENAI_API_BASE_URL=http://WHISPER_HOST:9000/v1
AUDIO_STT_OPENAI_API_KEY=replace-with-the-same-api-key
AUDIO_STT_MODEL=whisper-1
```

If Open WebUI and Whisper_not share a container network:

```env
AUDIO_STT_OPENAI_API_BASE_URL=http://whisper-not:9000/v1
```

If Open WebUI runs in Docker Desktop and Whisper_not publishes host port 9000:

```env
AUDIO_STT_OPENAI_API_BASE_URL=http://host.docker.internal:9000/v1
```

If Open WebUI runs on another machine:

```env
AUDIO_STT_OPENAI_API_BASE_URL=http://SERVER_LAN_IP:9000/v1
```

## Verify before configuring Open WebUI

```bash
curl http://WHISPER_HOST:9000/health
```

```bash
curl http://WHISPER_HOST:9000/v1/models \
  -H "Authorization: Bearer $WHISPER_API_KEY"
```

The model list includes `whisper-1`.

Test the exact response shape Open WebUI needs:

```bash
curl http://WHISPER_HOST:9000/v1/audio/transcriptions \
  -H "Authorization: Bearer $WHISPER_API_KEY" \
  -F file=@voice.webm \
  -F model=whisper-1
```

Expected:

```json
{"text":"..."}
```

## Language behavior

Recommended global setting:

```env
WHISPER_LANGUAGE=auto
```

When Open WebUI supplies a language, the request language overrides the global
setting. Otherwise faster-whisper detects the language.

The recommended German-optimized model remains multilingual, but recognition
quality outside its primary training distribution can vary.

## Optional features outside Open WebUI

Open WebUI’s normal speech-to-text path consumes only the `text` field. Use
the API directly for:

- word timestamps;
- speaker labels;
- SRT or VTT;
- downloadable responses;
- OpenAI-style diarized JSON.

This keeps voice input fast while preserving rich transcription features for
other applications.

## Troubleshooting

### Connection verification fails

- Confirm the API base URL ends in `/v1`.
- Confirm port 9000 is reachable from the Open WebUI container or host.
- Confirm the API keys match.
- Query `/v1/models` manually.

### HTTP 401

The bearer token does not match `WHISPER_API_KEY`.

### Empty or failed transcription

- Inspect `podman logs whisper-not` or `docker logs whisper-not`.
- Confirm the uploaded MIME type and extension are allowed by Open WebUI.
- Confirm the model finished downloading and `/health` returns `ok`.

### Open WebUI is slower than expected

- Keep `WHISPER_DIARIZATION=on_demand`.
- Keep `WHISPER_WORD_TIMESTAMPS=false`.
- Use CUDA with `WHISPER_COMPUTE_TYPE=float16`.
- Persist `/var/lib/whisper` so the model is not downloaded again.
