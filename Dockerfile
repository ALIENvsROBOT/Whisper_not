FROM python:3.12-slim

WORKDIR /opt/src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TMPDIR="/run/whisper-temp"

# faster-whisper decodes audio via the PyAV library (bundled FFmpeg libraries).
# curl is used by run.sh for the public-IP lookup and the health-check poll.
COPY ./requirements.txt /opt/src/requirements.txt

RUN set -x \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl bzip2 \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --requirement /opt/src/requirements.txt \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && find /opt/venv -name '*.pyi' -delete \
    && { find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true; } \
    && mkdir -p /var/lib/whisper

COPY ./run.sh /opt/src/run.sh
COPY ./manage.sh /opt/src/manage.sh
COPY ./api_server.py /opt/src/api_server.py
COPY ./diarizer.py /opt/src/diarizer.py
RUN chmod 755 /opt/src/run.sh /opt/src/manage.sh \
    && ln -s /opt/src/manage.sh /usr/local/bin/whisper_manage

EXPOSE 9000/tcp
VOLUME ["/var/lib/whisper"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD curl -fsS http://127.0.0.1:9000/health || exit 1
CMD ["/opt/src/run.sh"]

ARG BUILD_DATE
ARG VERSION
ARG VCS_REF
ENV IMAGE_VER=$BUILD_DATE

LABEL maintainer="ALIENvsROBOT <gowtham.sridher5@gmail.com>" \
    org.opencontainers.image.created="$BUILD_DATE" \
    org.opencontainers.image.version="$VERSION" \
    org.opencontainers.image.revision="$VCS_REF" \
    org.opencontainers.image.authors="ALIENvsROBOT <gowtham.sridher5@gmail.com>" \
    org.opencontainers.image.title="Whisper_not Speech-to-Text Server" \
    org.opencontainers.image.description="OpenAI-compatible faster-whisper server with opt-in word timestamps and speaker diarization." \
    org.opencontainers.image.url="https://github.com/ALIENvsROBOT/Whisper_not" \
    org.opencontainers.image.source="https://github.com/ALIENvsROBOT/Whisper_not" \
    org.opencontainers.image.documentation="https://github.com/ALIENvsROBOT/Whisper_not"
