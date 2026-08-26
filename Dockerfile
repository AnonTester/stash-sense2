# syntax=docker/dockerfile:1
#
# CPU-only variant (default/fallback for either GPU vendor). See
# Dockerfile.rocm (AMD, tested on this deployment) and Dockerfile.cuda
# (NVIDIA, best-effort/unverified) for GPU acceleration.

# Stage 0: Static ffmpeg (latest release)
FROM mwader/static-ffmpeg:latest AS ffmpeg-static

# Stage 1: Build dependencies
FROM python:3.11-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY requirements.docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.docker.txt

# Plain CPU onnxruntime -- insightface's own declared onnxruntime
# dependency (pulled in above) already resolves to this, so no ordering
# trick is needed here (only the GPU variants need one, since a
# vendor-specific build would otherwise get silently clobbered).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "onnxruntime>=1.17.0,<2"

# Stage 2: Runtime
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgles2 \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    # System ffmpeg (dynamically linked). Used by FFMPEG_HWACCEL=vaapi: the
    # static mwader build is fully static and cannot dlopen the libva
    # backend plugins VAAPI requires at runtime.
    ffmpeg \
    libva2 \
    libva-drm2 \
    mesa-va-drivers \
    && rm -rf /var/lib/apt/lists/*

# Static ffmpeg at /usr/local/bin/ffmpeg (takes PATH precedence over system
# ffmpeg). Used for CPU mode -- newer, better HEVC/10-bit support. VAAPI
# mode explicitly uses /usr/bin/ffmpeg (system package) instead.
COPY --from=ffmpeg-static /ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg-static /ffprobe /usr/local/bin/ffprobe

WORKDIR /app

COPY --from=build /app/venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

COPY api/ ./
# changelog.txt lives at repo root, not under api/ -- release_info.py
# reads it directly (both sidecar and plugin changelog sections) to
# answer "what's new" without any network call. Same cache-invalidation
# cost as the api/ layer above (changes on every version bump anyway).
COPY changelog.txt ./

RUN mkdir -p /data

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

STOPSIGNAL SIGTERM

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
