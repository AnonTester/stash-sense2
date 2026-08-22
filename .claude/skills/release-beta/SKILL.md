---
name: release-beta
description: Release a new beta version - bumps versions in all version files, commits, tags, and pushes
---

# Release Beta Version

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

Edit ALL FOUR files to the SAME new version:
- `api/main.py` - update `version="..."` in FastAPI app initialization
- `api/settings_router.py` - update `_version: str = "..."`
- `plugin/stash-sense.yml` - update `version:` field
- `plugin/stash-sense-core.js` - update `const PLUGIN_VERSION = '...'`

**CRITICAL**: All four files must have identical version strings.

### Step 2: Commit

```bash
git add api/main.py api/settings_router.py plugin/stash-sense.yml plugin/stash-sense-core.js
git commit -m "chore: bump version to X.Y.Z-beta.N"
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
1. Validates all version files match the tag (`scripts/check-version.sh`)
2. Builds 3 image variants (CPU default, AMD ROCm, NVIDIA CUDA) and pushes
   each to GitHub Container Registry under this repo's own namespace:
   `ghcr.io/anontester/stash-sense2`, `ghcr.io/anontester/stash-sense2-rocm`,
   `ghcr.io/anontester/stash-sense2-cuda` — tags `X.Y.Z-beta.N` and `beta`
3. Creates GitHub Release (marked as prerelease)

## Common Mistakes to Avoid

- Forgetting to update one of the four version files
- Tag doesn't match version (missing `v` prefix or typo)
- Pushing tag before pushing commit
- Creating tag on wrong branch
