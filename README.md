# Stash Sense

ML-powered performer identification and library curation for [Stash](https://github.com/stashapp/stash). Identifies performers in your scenes using face recognition, detects duplicate scenes, syncs upstream metadata changes, and surfaces actionable recommendations — all running locally on your hardware.

## What is Stash Sense?

Stash Sense is a sidecar service and Stash plugin that brings ML-powered analysis to your Stash library:

- **Face Recognition** — Identify performers in scenes and images using InsightFace's buffalo_l model. Matches against a database of 150,000+ performers sourced from StashDB, ThePornDB, and other stash-box endpoints, plus non-stash-box catalogue sources
- **Duplicate Scene Detection** — Find duplicate scenes using face fingerprints, stash-box IDs, and metadata overlap — catches duplicates that phash matching misses
- **Upstream Sync** — Detect metadata changes on stash-box endpoints and review per-field merge controls to keep your library current
- **Recommendations Dashboard** — A unified view of all suggestions: duplicates, unidentified scenes, missing stash-box links, and upstream updates
- **Self-Updating Database** — Check for and apply database updates from the Settings UI without restarting the container
- **Hardware-Adaptive** — Auto-detects your GPU and adjusts performance settings. Works with AMD (ROCm) or NVIDIA (CUDA) GPUs, or CPU-only (slower)

## Quick Start

There's no pre-built image to `docker pull` yet — build the sidecar locally from this repo with the Dockerfile matching your hardware, then install the plugin from the [stash-plugin-repo](https://github.com/AnonTester/stash-plugin-repo) index.

### Prerequisites

1. **Stash** running with scene sprite sheets generated
2. **Docker** with **Docker Compose** installed on your system
3. A GPU is optional — NVIDIA (CUDA) or AMD (ROCm) both work, or run CPU-only (slower, but the most portable and the least to go wrong)

### 1. Build and start the container

Clone this repo, then set your Stash connection details:

```bash
git clone https://github.com/AnonTester/stash-sense2.git
cd stash-sense2
cp api/.env.example .env
# edit .env: fill in STASH_URL and STASH_API_KEY
```

Pick the compose file matching your hardware and build+start:

| Hardware | Compose file | Dockerfile used | Status |
|----------|--------------|------------------|--------|
| CPU only | `docker-compose.yml` | `Dockerfile` | Tested, most portable |
| AMD GPU (ROCm) | `docker-compose.rocm.yml` | `Dockerfile.rocm` | Tested (reference deployment: Radeon 780M / gfx1103) |
| NVIDIA GPU (CUDA) | `docker-compose.cuda.yml` | `Dockerfile.cuda` | Best-effort, unverified — no NVIDIA hardware in the reference deployment |

```bash
# CPU
docker compose build && docker compose up -d

# AMD (ROCm) — needs the NVIDIA-equivalent ROCm userspace on the host,
# see GPU Troubleshooting below for /dev/kfd and /dev/dri access
docker compose -f docker-compose.rocm.yml build
docker compose -f docker-compose.rocm.yml up -d

# NVIDIA (CUDA) — needs the NVIDIA Container Toolkit on the host
docker compose -f docker-compose.cuda.yml build
docker compose -f docker-compose.cuda.yml up -d
```

Each variant listens on port `6960` and persists its data under `./api/data` — only run one at a time unless you also change the port/volume mappings to avoid a collision. First startup downloads the buffalo_l face recognition models on first use (or via Settings → Models → Download All once running) and can take a few minutes.

> **Want to try a different variant later?** `docker compose -f <other-file>.yml down` the one you're not using first — they all default to the same port and container data directory.

### 2. Verify it's running

```bash
curl http://localhost:6960/health
```

### 3. Install the Stash plugin

In Stash, go to **Settings > Plugins > Available Plugins**, click **Add Source**, and add:

| Field | Value |
|-------|-------|
| Name | Stash Sense |
| Source URL | `https://raw.githubusercontent.com/AnonTester/stash-plugin-repo/main/index.yml` |

The **Stash Sense** plugin will now show up in the Available Plugins list alongside any other plugins from that index — install it, then configure the sidecar URL (`http://your-host:6960`) in its settings.

To update later: **Settings > Plugins > Installed Plugins**, click **Check for Updates** (or reload the plugin source) — new plugin releases show up there the same way, whenever a new version lands in the [stash-plugin-repo index](https://github.com/AnonTester/stash-plugin-repo).

### 4. Download database and models

Navigate to `/plugins/stash-sense` in Stash to open the Stash Sense dashboard. From the **Settings** tab:

1. **Database** — Click **Update** to download the face recognition database (~150,000+ performers) from [stash-sense2-data](https://github.com/AnonTester/stash-sense2-data)
2. **Models** — Click **Download All** to download the required ONNX models (buffalo_l face recognition, ~200 MB; tattoo detection is optional and downloads separately if you enable that signal in Settings)

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASH_URL` | Yes | — | URL to your Stash instance (e.g., `http://stash:9999`) |
| `STASH_API_KEY` | Yes | — | Stash API key (Settings > Security > API Key) |
| `FFMPEG_HWACCEL` | No | `none` | ffmpeg hardware acceleration for frame extraction. See [ffmpeg Hardware Acceleration](#ffmpeg-hardware-acceleration) below. |

Additional performance and recognition settings are configurable via the **Settings** tab in the plugin UI. The sidecar auto-tunes defaults based on your hardware. Stash-box API keys are auto-discovered from your Stash instance's configured metadata providers (Settings > Metadata Providers).

### ffmpeg Hardware Acceleration

By default ffmpeg decodes video frames on the CPU. For very large or high-resolution files (4K+, 8K, 10-bit HEVC) this can exhaust container RAM and trigger the kernel OOM killer, crashing the container. Setting `FFMPEG_HWACCEL` offloads decoding to the GPU, keeping large frame buffers in GPU memory instead of host RAM.

| Value | GPU | Notes |
|-------|-----|-------|
| `none` | Any / CPU | Default in every compose file, regardless of which one you're running. |
| `cuda` | NVIDIA | Uses NVDEC. Only meaningful with `docker-compose.cuda.yml`, which already reserves the GPU it needs. |
| `vaapi` | AMD or Intel | Uses VAAPI. `docker-compose.rocm.yml` already maps `/dev/dri` and defaults to this; for CPU-inference + VAAPI-decode on the base compose file, uncomment its `devices:` block instead. |

**Trade-off:** hwaccel reduces memory pressure but adds a small per-frame GPU context setup cost, which slightly slows down single-frame seeks. For normal 1080p/4K libraries the default CPU mode is faster. Use hwaccel only if you are hitting OOM crashes on specific large files.

**NVIDIA (CUDA):**

Set `FFMPEG_HWACCEL=cuda` in `docker-compose.cuda.yml`'s `environment:` section — the GPU reservation it needs is already there.

**AMD / Intel (VAAPI):**

`docker-compose.rocm.yml` already maps `/dev/dri` and defaults to `FFMPEG_HWACCEL=vaapi`. On the CPU compose file, uncomment the `devices:` block and set the env var yourself:
```yaml
environment:
  - FFMPEG_HWACCEL=vaapi
devices:
  - /dev/dri/renderD128:/dev/dri/renderD128
```

Check your host's render node with `ls /dev/dri/` — use `renderD129` if you have multiple GPUs and `renderD128` is your primary display adapter.

> **If VAAPI fails** (ffmpeg logs "Device creation failed" or "No device available for decoder"), the affected scenes are marked as errors and skipped on crash-recovery reruns. To retry them, start a **new** fingerprint generation job from the Operations tab — new jobs always retry error scenes while skipping already-completed ones.

## Updating

### Database Updates

Stash Sense checks for new database releases automatically. To update:

1. Open the **Settings** tab in the plugin
2. Check the **Database** section for available updates
3. Click **Update** — the sidecar downloads and hot-swaps the data without restarting

### Container Updates

No pre-built image to pull yet — pull the latest source and rebuild with the same compose file you started with (data under `./api/data` and the named `insightface` volume both persist across a rebuild):

```bash
git pull
docker compose build && docker compose up -d          # CPU
docker compose -f docker-compose.rocm.yml build && docker compose -f docker-compose.rocm.yml up -d   # AMD
docker compose -f docker-compose.cuda.yml build && docker compose -f docker-compose.cuda.yml up -d   # NVIDIA
```

Your recommendation history and settings are stored separately from the face database and persist across both types of updates.

## Documentation

This README is the current source of truth for setup and configuration. The `docs/` folder in this repo is inherited from the upstream project this was forked from and hasn't been fully updated for the buffalo_l/usearch migration yet — treat it as unreliable until that's done.

## Requirements

| Component | Requirement |
|-----------|-------------|
| Stash | v0.25+ with sprite sheets generated |
| Docker | With Docker Compose; plus `nvidia-container-toolkit` (NVIDIA) or a working ROCm host install (AMD), only if using GPU acceleration |
| GPU | Optional — AMD (ROCm, tested) or NVIDIA (CUDA, best-effort) with 4GB+ VRAM recommended; CPU-only fallback works for every feature, just slower for face recognition |
| Disk | ~1.5 GB for the face recognition database + models, plus working space for frame extraction |

## GPU Troubleshooting

**NVIDIA (CUDA):** requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.

| Problem | Solution |
|---------|----------|
| `docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]` | Install `nvidia-container-toolkit` and restart Docker |
| GPU not detected inside container | Verify with `nvidia-smi` on the host; ensure the toolkit is configured: `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker` |
| unRAID | Add `--runtime=nvidia --gpus all` in **Extra Parameters** on the Docker container config page |

**AMD (ROCm):** `docker-compose.rocm.yml` maps `/dev/kfd` and `/dev/dri` and sets `security_opt: seccomp=unconfined` — no separate toolkit install needed beyond the host having a working kernel driver for your card (`rocminfo` should list your GPU). If your card isn't on ROCm's officially supported list (e.g. integrated/APU parts like the Radeon 780M this was tested on), you may need `HSA_OVERRIDE_GFX_VERSION` set to the nearest supported target — check `rocminfo`'s reported `gfx` version. This is a compose-level environment variable (`docker-compose.rocm.yml`'s `HSA_OVERRIDE_GFX_VERSION`, defaulting to the reference deployment's `11.0.0`), not baked into the image, so set it to match *your* card (or delete the line if your card doesn't need an override) without needing to rebuild.

**Neither?** Use the default `docker-compose.yml` (CPU-only) — every feature works, face recognition is just slower per scene.

## Support

- **Bug Reports**: [GitHub Issues](https://github.com/AnonTester/stash-sense2/issues)

## License

MIT License — See [LICENSE](LICENSE) for details.
