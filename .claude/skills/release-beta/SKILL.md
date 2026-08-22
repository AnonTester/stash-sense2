---
name: release-beta
description: Release a new sidecar beta version - bumps the sidecar's version files, commits, tags, and pushes to trigger the GHCR image build
---

# Release Sidecar Beta Version

This releases the **sidecar** (the Docker image built and published to GHCR). Plugin releases are a separate, independent track published through `AnonTester/stash-plugin-repo` — see `stash-sense-context`'s "Related Skills" note. Don't bump plugin files here unless the sidecar release is bundled with a plugin change too (in which case bump the plugin pair to whatever version makes sense for it, not necessarily matching this tag).

## Version Convention

Beta versions follow this pattern: `X.Y.Z-beta.N`

- Increment beta number (e.g., `0.1.0-beta.3` -> `0.1.0-beta.4`)
- For new minor/patch, reset to beta.1 (e.g., `0.1.0-beta.3` -> `0.1.1-beta.1`)

## Pre-Flight Checks

1. Ensure you're on `main` branch and it's up to date
2. Check current version: look at `api/main.py` FastAPI `version=` field
3. Determine next version based on convention above

## Release Steps

### Step 1: Update Versions

Edit both sidecar version files to the SAME new version:
- `api/main.py` - update `version="..."` in FastAPI app initialization
- `api/settings_router.py` - update `_version: str = "..."`

**CRITICAL**: Both files must have identical version strings — `scripts/check-version.sh` (run by CI on tag push) enforces this.

### Step 2: Commit

```bash
git add api/main.py api/settings_router.py
git commit -m "chore: bump sidecar version to X.Y.Z-beta.N"
```

### Step 3: Push to Main

```bash
git push origin main
```

### Step 4: Create and Push Tag

The tag MUST match the version with a `v` prefix:

```bash
git tag vX.Y.Z-beta.N
git push origin vX.Y.Z-beta.N
```

**Example**: Version `0.1.0-beta.1` -> Tag `v0.1.0-beta.1`

## What Happens Next

GitHub Actions (`.github/workflows/docker-build.yml`) triggers on tag push:
1. Validates the sidecar version pair matches the tag (`scripts/check-version.sh`) — the plugin pair is checked for internal consistency only, not against the tag
2. Builds 3 image variants (CPU default, AMD ROCm, NVIDIA CUDA) and pushes
   each to GitHub Container Registry under this repo's own namespace:
   `ghcr.io/anontester/stash-sense2`, `ghcr.io/anontester/stash-sense2-rocm`,
   `ghcr.io/anontester/stash-sense2-cuda` — tags `X.Y.Z-beta.N` and `beta`
3. Creates GitHub Release (marked as prerelease)

## Common Mistakes to Avoid

- Forgetting to update one of the two sidecar version files
- Tag doesn't match version (missing `v` prefix or typo)
- Pushing tag before pushing commit
- Creating tag on wrong branch
