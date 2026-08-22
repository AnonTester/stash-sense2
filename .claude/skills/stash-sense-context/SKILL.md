---
name: stash-sense-context
description: Use when starting work on stash-sense2 to load project context, architecture overview, and development patterns. Reference for any implementation decisions.
---

# Stash Sense 2 Project Context

Quick-reference for architecture, conventions, and operational knowledge. This repo is a fork of the original `carrotwaxr/stash-sense` project, migrated from a legacy RetinaFace/FaceNet512+ArcFace/Voyager pipeline to InsightFace's `buffalo_l` bundle + a single usearch index — architecture below reflects the current, post-migration state only.

## Architecture

**Two components:**
- **Sidecar API** (`api/`) — Python/FastAPI, face recognition, recommendations, upstream sync
- **Plugin** (`plugin/`) — JS/CSS/Python injected into Stash web UI, proxies to sidecar

**Two databases:**
- `performers.db` — Read-only, distributed via GitHub Releases on `AnonTester/stash-sense2-data`. Face metadata, stash-box IDs, a single usearch ANN index. Built by the separate, private `AnonTester/stash-sense2-data-gen` repo.
- `stash_sense.db` — Read-write, user-local. Recommendations, watermarks, upstream snapshots, fingerprints, job queue.

**Deployment:**
- Sidecar: Docker container — 3 published variants (CPU default, AMD ROCm, NVIDIA CUDA best-effort), see `docker-compose*.yml`/`Dockerfile*`. Published to `ghcr.io/anontester/stash-sense2[-rocm|-cuda]` on version-tag push (`.github/workflows/docker-build.yml`).
- Plugin: installed as a local Stash plugin; `PLUGIN_ID` (in `plugin/stash-sense-core.js`) must match the install folder's directory name exactly, or it silently reads/drives whichever other plugin owns that id (see CLAUDE.md's "Plugin identity" note).
- Public plugin distribution: `AnonTester/stash-plugin-repo`'s index (separate from this repo's own GHCR image publishing — see CLAUDE.md's "Publishing to the public plugin index").
- Dev sidecar: `http://localhost:5000` with `--reload` (`cd api && make sidecar`)

## Key Systems

| System | Key Files | Pattern |
|--------|-----------|---------|
| Face Recognition | `embeddings.py`, `recognizer.py`, `matching.py`, `face_config.py` | Single buffalo_l detect+embed call per image, one usearch index, plain nearest-neighbor + threshold filter |
| Recommendations | `recommendations_router.py`, `recommendations_db.py`, `analyzers/` | `BaseAnalyzer` + incremental watermarking |
| Duplicate Detection | `analyzers/duplicate_scenes.py`, `analyzers/duplicate_performer.py`, `analyzers/duplicate_scene_files.py` | Candidate generation (SQL joins) -> sequential scoring |
| Upstream Sync | `upstream_field_mapper.py`, `stashbox_client.py`, `analyzers/base_upstream.py`, `analyzers/upstream_*.py` | 3-way diff (upstream vs local vs snapshot), logic versioned. `UpstreamSceneAnalyzer._diff_fields()` deliberately returns a single dict (relational diff), not `list[dict]` like every other entity type — don't assume list shape in shared `base_upstream.py` code paths. |
| Job Queue | `queue_manager.py`, `job_models.py` (`JOB_REGISTRY`), `base_job.py` | Resource-slot-limited (`ResourceType.GPU/CPU_HEAVY/NETWORK/LIGHT`), cursor-based checkpointing, stop signaling |
| Hardware-Adaptive Settings | `hardware.py`, `settings.py` (`TIER_DEFAULTS`) | One-shot startup probe (NVIDIA via pynvml, AMD via rocminfo + amdgpu sysfs) picks a tier (`gpu-high`/`gpu-low`/`cpu`) that drives batch size/concurrency defaults; `num_frames`/`detection_size` deliberately NOT tier-varied (accuracy, not just speed) |
| Gallery ID | `recognizer.py` (`/identify/image`, `/identify/gallery`) | Independent images, aggregate by performer |
| DB Self-Update | `database_updater.py` | download -> verify -> swap -> reload, delta-chain support, 503 gating |
| Model Downloads | `model_manager.py`, `models.json` | buffalo_l ONNX files downloaded at runtime from GitHub Releases (not baked into the image) |
| Plugin Proxy | `stash_sense_backend.py` | All sidecar calls go through plugin backend to bypass browser CSP |

## Development Commands

```bash
# Start sidecar (dev)
cd api && make sidecar

# Run tests
cd api && make test        # or: make test-ci (no ML/GPU deps, matches CI), make test-heavy

# Lint
cd api && make lint        # or: make lint-fix

# Check sidecar/plugin version pairs are each internally consistent (see CLAUDE.md)
./scripts/check-version.sh
```

## Conventions

- **Logging:** Default level is WARNING. Use `logger.warning()` for user-visible progress. `logger.info()` is not visible.
- **Rate limiting:** Shared 5 req/s for Stash and StashBox APIs. StashBox uses `Priority.LOW`.
- **Plugin defaults:** Plugin sends NO face recognition defaults; relies on sidecar `face_config.py`.
- **Background tasks:** Don't inherit shell activation. Use explicit venv python path for background processes.
- **Hot reload caveat:** Background analysis tasks block uvicorn `--reload` on file changes; must kill and restart.
- **Plugin logging:** Use Stash's log protocol with level prefix bytes (`\x01` + level_char + `\x02`), not plain JSON to stderr. See `stash_sense_backend.py:_log_prefix()`.
- **Upstream logic versioning:** Each upstream analyzer has a `logic_version` class attribute. When bumped, the next analysis run auto-clears stale snapshots and watermarks, forcing full re-analysis. Bump when comparison logic changes (field sets, normalization, ID resolution).
- **Local-only fields:** `favorite`, `rating`, `o_count` are local Stash metadata — never compare against upstream StashBox values.
- **Resolved/dismissed recommendations sort by recency, not confidence** (`recommendations_db.py:get_recommendations()`) — only pending items are confidence-ranked; once acted on, most-recently-touched should surface first for review.

## Field Mapping (Stash vs StashBox)

| Diff Engine | Stash Mutation | StashBox | Notes |
|---|---|---|---|
| `aliases` | `alias_list` | `aliases` | |
| `height` | `height_cm` (Int) | `height` (Int) | |
| `breast_type` | `fake_tits` | `breast_type` | |
| `career_start_year` | `career_length` (String) | `career_start_year` (Int) | Combined "YYYY-YYYY" |
| `career_end_year` | `career_length` (String) | `career_end_year` (Int) | Combined "YYYY-YYYY" |
| `cup_size` | `measurements` (String) | `cup_size` | Combined "38F-24-35" |
| `band_size` | `measurements` (String) | `band_size` (Int) | Combined "38F-24-35" |
| `waist_size` | `measurements` (String) | `waist_size` (Int) | Combined "38F-24-35" |
| `hip_size` | `measurements` (String) | `hip_size` (Int) | Combined "38F-24-35" |

Translation: `recommendations_router.py:update_performer_fields()`

## Face Recognition Tuned Defaults (`face_config.py`)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `MAX_DISTANCE` | 0.5 | Legacy (FaceNet512+ArcFace fusion) tuned value, carried over as a placeholder — NOT re-validated for buffalo_l's single-embedding distance space yet. Needs its own re-tuning pass. |
| `NUM_FRAMES` | 60 | Same across all hardware tiers — deliberately not lowered for CPU (accuracy tradeoff, not just speed) |
| `MIN_FACE_SIZE` / `MIN_FACE_CONFIDENCE` | 40px / 0.5 | Detection thresholds |
| `CLUSTER_THRESHOLD` | 0.6 cosine distance | Face clustering (single buffalo_l embedding space, not concatenated dual-model) |
| `TOP_K` | 3 | Top matches per person |
| `SPRITE_MAX_FRAMES` | 300 | Sprite-sheet tiles are cheap (no decode/seek) relative to `NUM_FRAMES`, so this is a pathological-case cap, not a cost/accuracy tradeoff |

`EMBEDDING_DIM` (`config.py`) = 512 (single buffalo_l embedding, replacing the old separate `FACENET_DIM`/`ARCFACE_DIM`).

## Docker Images

3 variants, see each `Dockerfile*` for hardware-specific build notes:
- `Dockerfile` — CPU-only, `python:3.11-slim` base. Default/fallback for either GPU vendor.
- `Dockerfile.rocm` — AMD ROCm, `python:3.11-slim` base + AMD's apt repo. The variant actually run and tested (Radeon 780M / gfx1103) — see README's GPU Troubleshooting section for the `HSA_OVERRIDE_GFX_VERSION`/`amdgpu.cwsr_enable` gotchas this hardware needed.
- `Dockerfile.cuda` — NVIDIA CUDA, `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` base. Best-effort/unverified — no NVIDIA hardware in the reference deployment.

buffalo_l ONNX models are downloaded at runtime (`model_manager.py`), not baked into any image. Port: 5000 (container) — host mapping is a user/deployment choice, not fixed. Volumes: `/data` (usearch index, `performers.db`, `stash_sense.db`), `/root/.insightface` (detector cache).

## GitHub

- Repo: `AnonTester/stash-sense2` (public — see CLAUDE.md's Public Repository Policy)
- Data repo (public): `AnonTester/stash-sense2-data` — GitHub Releases host `performers.db` + usearch index + models
- Data-gen repo (private): `AnonTester/stash-sense2-data-gen` — builds `performers.db`, crawls stash-box endpoints, publishes releases
- Public plugin index: `AnonTester/stash-plugin-repo`

## Related Skills (this repo)

- `release-beta` — cut a sidecar beta release (bumps the 2 sidecar version files, tag, push)
- `release-stable` — cut a stable version release
- `db-import-export` — copy face recognition data from the data-gen pipeline to the sidecar (note: still describes the pre-migration dual-Voyager-index file set as of writing — verify current file names in `config.py`'s `DatabaseConfig` before following it literally)
