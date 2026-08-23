---
name: release-stable
description: Release a stable sidecar version (non-beta) - bumps the sidecar's version files, commits, tags, and pushes to trigger the GHCR image build
---

# Release Stable Sidecar Version

Use this skill to release a stable (non-beta) **sidecar** version after beta testing is complete. This releases the Docker image built and published to GHCR. Plugin releases are a separate, independent track published through `AnonTester/stash-plugin-repo` — see `stash-sense-context`'s "Related Skills" note. Don't bump plugin files here unless the sidecar release is bundled with a plugin change too (in which case bump the plugin pair to whatever version makes sense for it, not necessarily matching this tag).

**Only run this when the user explicitly asks for a release** — not routinely after a change, even a sidecar one. The tag push triggers a real GHCR image build (all 3 variants, ~30min for the ROCm one) and a public GitHub Release; see CLAUDE.md's "Cutting a sidecar release tag" note.

## Version Convention

Stable versions follow semantic versioning: `X.Y.Z`

- Remove beta suffix when promoting: `0.1.0-beta.8` -> `0.1.0`
- Or increment version for new release: `0.1.0` -> `0.1.1` or `0.2.0`

## Pre-Flight Checks

1. Ensure you're on `main` branch and it's up to date
2. Check current version: look at `api/main.py` FastAPI `version=` field
3. Determine target version (remove beta suffix or increment)

## Release Steps

### Step 1: Update Versions

Edit both sidecar version files to the SAME new version:
- `api/main.py` - update `version="..."` in FastAPI app initialization
- `api/settings_router.py` - update `_version: str = "..."`

**CRITICAL**: Both files must have identical version strings — `scripts/check-version.sh` (run by CI on tag push) enforces this.

### Step 2: Commit

```bash
git add api/main.py api/settings_router.py
git commit -m "chore: bump sidecar version to X.Y.Z"
```

### Step 3: Push to Main

```bash
git push origin main
```

### Step 4: Create and Push Tag

The tag MUST match the version with a `v` prefix:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

**Example**: Version `0.1.0` -> Tag `v0.1.0`

## What Happens Next

GitHub Actions (`.github/workflows/docker-build.yml`) triggers on tag push:
1. Validates the sidecar version pair matches the tag (`scripts/check-version.sh`) — the plugin pair is checked for internal consistency only, not against the tag
2. Builds 3 image variants (CPU default, AMD ROCm, NVIDIA CUDA) and pushes
   each to GitHub Container Registry under this repo's own namespace:
   `ghcr.io/anontester/stash-sense2`, `ghcr.io/anontester/stash-sense2-rocm`,
   `ghcr.io/anontester/stash-sense2-cuda` — tags `X.Y.Z`, `X.Y`, `latest`
3. Creates GitHub Release with auto-generated notes

## Semantic Versioning Guide

- **Patch** (0.1.X): Bug fixes, minor tweaks, no new features
- **Minor** (0.X.0): New features, backward-compatible changes
- **Major** (X.0.0): Breaking changes (API overhauls, major architecture changes)

## Common Mistakes to Avoid

- Forgetting to update one of the two sidecar version files
- Tag doesn't match version (missing `v` prefix or typo)
- Pushing tag before pushing commit
- Creating tag on wrong branch
- Leaving beta suffix in version string
