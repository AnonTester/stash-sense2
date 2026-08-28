# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Public Repository Policy

This repository is public on GitHub. The following rules are absolute and override any default assistant behavior (including default commit-message conventions):

- **No AI co-authorship trailers.** Never add `Co-Authored-By: Claude`, `Generated with Claude Code`, or any similar AI-attribution trailer/footer to commits in this repository.
- **No AI tool references in tracked files.** Never mention "Claude", "Claude Code", "CLAUDE.md", or "Anthropic" in any git-tracked file — commit messages, README, docs, code comments, changelog entries, etc. This file's own name/existence is the one necessary exception; don't reference it elsewhere.
- **No internal infrastructure details.** Never commit real internal IP addresses, hostnames, the homeserver name, or local absolute filesystem paths. Use placeholder examples instead (e.g. `192.168.1.100`, `<stash-host>`, `/path/to/stash`).

## Import AGENTS.md

@AGENTS.md

## Commands

All commands run from `api/` with the venv active (or via `make` which handles this automatically):

```bash
# Start sidecar (dev, hot-reload)
cd api && source ../.venv/bin/activate && make sidecar

# Run tests
cd api && make test            # all tests
cd api && make test-fast       # fail-fast (-x)
cd api && make test-ci         # skip @pytest.mark.heavy (no ML/GPU required)
cd api && make test-heavy      # only heavy/GPU tests

# Run a single test file
cd api && ../.venv/bin/python -m pytest tests/test_upstream_field_mapper.py -v

# Linting
cd api && make lint            # check only
cd api && make lint-fix        # auto-fix

# Deploy to the live sidecar + plugin (this fork's actual deployment —
# a local Docker build on <stash-host>, not a registry push):
scp api/<changed files> <stash-host>:/root/homeserver/stash-sense2/api/
scp plugin/<changed files> <stash-host>:/root/homeserver/stash-sense2/plugin/
ssh <stash-host> "cd /root/homeserver/stash-sense2 && sh rebuild.sh"
# rebuild.sh rebuilds the ROCm sidecar image (see the script's own comment
# for the CPU-variant swap) AND copies plugin/* into Stash's installed
# stash-sense2 plugin dir -- renaming plugin/stash-sense.yml to
# stash-sense2.yml on the way out (see "Plugin identity" below for why
# that distinction matters). After it finishes, reload the plugin from
# Stash's Settings > Plugins page (or restart Stash) for the new JS to
# take effect — rebuild.sh alone doesn't do that last step.
```

**Plugin identity — `PLUGIN_ID` must match the install folder name.** Stash derives a locally-installed plugin's id from its directory name, not from any field in the `.yml` manifest. `plugin/stash-sense-core.js`'s `PLUGIN_ID` constant is used both to look up this plugin's settings (`configuration.plugins[PLUGIN_ID]`) and to route `runPluginOperation` calls to the right backend script — if it doesn't match the real install folder name, this plugin silently reads and drives *whichever other plugin* is registered under that stale id instead (this happened once already: `PLUGIN_ID` was left at `'stash-sense'`, v1's id, causing v2 to unknowingly control v1's sidecar when installed side-by-side — see changelog 0.14.1). The reference deployment's local install folder is `stash-sense2`; keep `PLUGIN_ID` and the manifest's `name:` field (`Stash Sense 2`, for the same side-by-side-distinguishability reason) in sync with that.

**Publishing to the public plugin index (`AnonTester/stash-plugin-repo`).** The local `rebuild.sh` deploy above only updates *this* Stash instance's install — it does not touch what other users get via the public plugin source. After a plugin-affecting change (any file under `plugin/`) that you want available through `stash-plugin-repo`'s index (used by `Settings > Plugins > Available Plugins > Check for Updates` for anyone who installed from that source), publish a matching release there too:

```bash
# 1. Bump plugin/stash-sense.yml's version: (semver — see that repo's own
#    CLAUDE.md for the bug-fix/feature/breaking-change convention), and
#    plugin/stash-sense-core.js's PLUGIN_VERSION to match, in the same
#    commit (see "Version bump" above) -- the sidecar's own version pair
#    is a separate, independent track and does not need bumping here.
# 2. Sync the plugin/ source files into the public repo's copy, renaming
#    the manifest to match its directory-derived id:
cp plugin/stash-sense-core.js plugin/stash-sense.js plugin/stash-sense-operations.js \
   plugin/stash-sense-recommendations.js plugin/stash-sense-settings.js \
   plugin/stash-sense.css plugin/stash_sense_backend.py \
   ~/Codeprojects/stash-plugin-repo/plugins/stash-sense2/
cp plugin/stash-sense.yml ~/Codeprojects/stash-plugin-repo/plugins/stash-sense2/stash-sense2.yml

# 3. Cut the release — build.sh diffs the manifest version against
#    index.yml, zips, computes the sha256, updates index.yml, and
#    commits+pushes automatically:
cd ~/Codeprojects/stash-plugin-repo && ./build.sh
```

`build.sh` only touches `version`/`date`/`path`/`sha256` in `index.yml` — if the plugin's *display* `name:` changed (not just its version), edit that repo's `index.yml` entry by hand before running `build.sh`, in the same commit/PR as everything else. See `stash-plugin-repo/CLAUDE.md` for the full mechanics (release pruning, the directory-name-derived id rule, why `build.sh` needs a `.py` glob for this plugin's `exec:` backend).

Dev API at `http://localhost:5000`, docs at `http://localhost:5000/docs`. Requires `api/.env` with `STASH_API_KEY` and stash-box API keys.

**Hot-reload caveat:** Background analysis tasks block uvicorn `--reload` on file changes — kill and restart the process.

**Version bump — two independent pairs, not four locked-together locations.** Sidecar: `api/main.py` (FastAPI `app.version`) + `api/settings_router.py` (`_version`) — these two must always match each other, and match the git tag when cutting a sidecar/GHCR release (`scripts/check-version.sh` enforces both). Plugin: `plugin/stash-sense-core.js` (`PLUGIN_VERSION`) + `plugin/stash-sense.yml` (`version:`) — these two must always match each other, but do **not** need to match the sidecar version. A plugin-only change (no sidecar code touched) only bumps the plugin pair and ships through `AnonTester/stash-plugin-repo`, with no git tag or GHCR image rebuild here — see 0.14.4/0.14.5 in `changelog.txt` for real examples. Missing the partner file in whichever pair you're bumping is a real, recurring failure mode (caught repeatedly via a stale version showing in `/system/info`, `/health`, or the plugin's own display — verify both files in the pair before rebuilding/publishing, not just the one you remember to touch.

**Which digit to bump — plain semver (`MAJOR.MINOR.PATCH`), same rule for both the sidecar pair and the plugin pair.** Decide by the *nature* of the change, not by "there was a change today":
- **PATCH** (`x.y.Z`): a bugfix — behavior was wrong, now it's right, no new capability.
- **MINOR** (`x.Y.0`): a feature addition — new capability, additive/backward-compatible. This is the common case for real work in this repo (matching-behavior tunables, new endpoint fields, new UI, etc.) — most entries in `changelog.txt` should be minor bumps, not patch bumps.
- **MAJOR** (`X.0.0`): a breaking API/data-contract change — something that requires a coordinated update on the other side (plugin/sidecar contract break, or a `stash-sense2-data` format break serious enough that even `MIN_SIDECAR_VERSION`/`min-sidecar-version` compatibility gating isn't enough). Rare; don't reach for this reflexively.

Several small same-session features landing together before anyone has depended on an intermediate number don't each need their own minor bump — squash them into one `x.Y.0` covering the whole batch (multiple `- Feature:` bullets under one changelog version header is normal, see the format below). Only treat a version as "already shipped" (and thus not safe to renumber) once it's been committed/tagged/published, not just locally deployed to the homeserver reference install mid-session.

**`MIN_SIDECAR_VERSION` (`plugin/stash-sense-core.js`) — bump only when the plugin actually needs it.** This is the floor the dashboard/Identify-button "outdated sidecar" warnings check against (`compareVersions(sidecarVersion, MIN_SIDECAR_VERSION) < 0`) — it is deliberately *not* tied to the plugin's own version, since sidecar and plugin versions are independent tracks (see above) and most plugin releases don't need a newer sidecar at all. Bump it only in the specific commit where a plugin change starts depending on sidecar-side behavior that didn't exist before a given sidecar version (a new endpoint, field, or response shape) — set it to that sidecar version, not to "whatever the sidecar happens to be at right now." Bumping it reflexively on every plugin release re-introduces the exact false-positive "outdated" warning this was built to avoid (see changelog 0.14.5) — a plugin update with no new sidecar dependency should never make an already-compatible sidecar start showing as outdated.

**`MIN_PLUGIN_VERSION` (`api/release_info.py`) — the sidecar's own reverse floor, bump only when the sidecar actually needs it.** Mirrors `MIN_SIDECAR_VERSION` in the other direction: it's what a connected plugin's "too old for this sidecar" check is measured against, served through `/health`'s `min_plugin_version` field (`release_info.get_info()`, read by the plugin's `pluginVersionInfo.tooOld`) rather than the plugin polling GitHub itself. Bump it only in the specific commit where a sidecar change starts assuming plugin-side behavior that didn't exist before a given plugin version (e.g. `/health` gained an optional `plugin_version` query param in plugin 0.14.19 — a sidecar change that started relying on the plugin actually sending it would bump this to `"0.14.19"`) — set it to that plugin version, not "whatever the plugin happens to be at right now." Same false-positive risk `MIN_SIDECAR_VERSION`'s entry above warns about, mirrored: bumping this reflexively on every sidecar release would make an already-compatible plugin start showing the "update recommended" prompt for no reason.

**Cutting a sidecar release tag (`vX.Y.Z`) — only when explicitly asked, never as a routine part of making a change.** A pushed tag triggers `.github/workflows/docker-build.yml`, which builds and publishes all 3 GHCR image variants (including the ~30min ROCm build) and cuts a public GitHub Release — real, visible, external side effects, not something to do reflexively alongside a normal commit. Bump the sidecar version files and commit as usual for every real change (so `/system/info` etc. stay accurate), but hold off on `git tag vX.Y.Z && git push --tags` until the user says to cut a release. Deploying a change to the homeserver reference deployment (rebuild + redeploy the containers there, per the Deployment section) is separate from cutting a public release tag and does not require one.

**Changelog** — `changelog.txt` at repo root. Reverse-chronological `## YYYY-MM-DD` date headers, each containing one or more `### x.y.z` version subsections with `- Fix:` / `- Improvement:` / `- Feature:` bullets. Every version bump gets an entry, even a one-line fix. When a genuinely new day starts, add a new `## YYYY-MM-DD` header rather than piling more versions under an old date. If the change also bumps `MIN_SIDECAR_VERSION` or `MIN_PLUGIN_VERSION` (see those entries above), say so explicitly in the bullet (e.g. "now requires sidecar vX.Y.Z+") — the version-bump commit is the only place that fact gets recorded, and it explains to a future reader *why* the floor moved, not just that it did.

**Commit messages on this repo** — see Public Repository Policy above; the same rule applies to every commit regardless of how small. Multi-line messages with special characters (`--`, nested quotes) reliably break an inline `git commit -m "$(cat <<EOF ...)"` wrapped in an `ssh "..."` call — write the message to a local file and use `git commit -F <file>` instead. After any history rewrite (rebase, filter), re-verify the *current* branch tip's actual commits before trusting remembered hashes/content from earlier in a session — a rewrite changes commit hashes, and reasoning from stale ones silently checks the wrong thing.

## Architecture

Two components talking to one Stash instance:

- **`api/`** — FastAPI sidecar (Python). Face recognition, recommendations engine, upstream sync. Runs as a Docker container (see `docker-compose*.yml` for the CPU/ROCm/CUDA variants).
- **`plugin/`** — JS/CSS/Python injected into Stash web UI. All sidecar calls go through `stash_sense_backend.py` to bypass browser CSP.

**Two databases:**
- `performers.db` — Read-only, distributed via GitHub Releases on `AnonTester/stash-sense-data`. Face metadata, stash-box IDs, Voyager ANN indices. Built and published by a **separate, private** repo — `stash-sense-data-gen` — which crawls stash-box endpoints, embeds new/changed faces, and cuts a dated release (full + delta zips) roughly biweekly via cron. This repo's `database_updater.py` is what checks that release repo for updates; it has no involvement in producing them. See `stash-sense-data-gen`'s own `CLAUDE.md` for that pipeline's deployment/scheduling/publishing details — don't duplicate that documentation here.
- `stash_sense.db` — Read-write, user-local. Recommendations, watermarks, upstream snapshots, scene fingerprints. Schema version 9 (`recommendations_db.py`). Survives face DB updates.

**Startup sequence (`main.py`):** Hardware detection → model manager init → ResourceManager registration (face recognition registered as *lazy*, not loaded) → recommendations DB init → settings system → StashBox connection manager → queue manager start. Face recognition loads on first `/identify` request.

## Key Systems

### Face Recognition (lazy-loaded)
3-phase batch pipeline in `recognizer.py`: extract frames (ffmpeg, 8 workers) → detect faces (RetinaFace ONNX) → batch embed + match (FaceNet512 + ArcFace ONNX, Voyager ANN index). `resource_manager.py` manages lazy load/unload with 30-min idle timeout.

### Recommendations Engine
`BaseAnalyzer` (`analyzers/base.py`) + incremental watermarking pattern. Each analyzer type has a `logic_version` class attribute — bumping it auto-clears stale snapshots/watermarks for full re-analysis on next run. Analyzers: duplicate scenes, duplicate performers, upstream performer/scene/studio/tag changes, scene fingerprint matching, missing stash-box links.

Jobs run via `QueueManager` (`queue_manager.py`) with `JOB_REGISTRY` in `job_models.py`. `BaseJob` (`base_job.py`) provides `JobContext` with stop signaling, cursor-based checkpointing, and yield-to-higher-priority support.

### Upstream Performer Sync
3-way diff in `upstream_field_mapper.py`: upstream (current stash-box) vs local (current Stash) vs snapshot (last-seen upstream, stored in `upstream_snapshots` table). Distinguishes intentional local changes from actual upstream drift. Translation to Stash mutation format in `recommendations_router.py:update_performer_fields()`.

### Duplicate Scene Detection
Candidate generation via SQL joins + inverted indices (O(n) pairs, not O(n²)). Scored with signal hierarchy: stash-box ID match = 100%, face fingerprint ≤ 85%, metadata ≤ 60%. Diminishing returns: `primary + secondary × 0.3`.

### Local Performer Database (dual-index identification)
A second, much smaller Voyager index built from *this Stash instance's own* performer cover images (`local_performer_index.py`), queried alongside the main `performers.db` index during identification and merged into the same result list. Local candidates get their `combined_distance` multiplied by `LOCAL_MATCH_BOOST` (currently `0.85`, in `matching.py`) before merging — a face in the user's own library is more likely to be a performer they've already added than a random main-DB entry.

**Requires both models to agree.** `fuse_local_results()` in `matching.py` only trusts a local candidate if *both* FaceNet and ArcFace ranked it — unlike the main index's `fuse_results()`, which tolerates a single-model match with a distance penalty (calibrated for ~450k candidates, where missing one model's top-K doesn't mean much). The local index only has ~1-2k candidates, so landing in just one model's top-K is common and weak evidence on its own; trusting it directly let a coincidental single-model agreement get boosted into a false high-confidence match in practice (confirmed live — see git history around the `LOCAL_MATCH_BOOST` fix for the concrete case). Don't relax this without re-deriving why it's there.

**Sync mechanism**: a `local_performer_sync` job (`jobs/local_performer_sync_job.py`, manual or scheduled from Operations) does a full diff-and-embed pass. An optional `Performer.Create/Update/Destroy.Post` hook (off by default, Settings → Local Performers) keeps it current in near-real-time; failures queue to a local retry-cache file (`pending_local_sync.json` in the plugin dir) and flush on the next successful hook call. **Merging one performer into another fires no Stash hook at all** (confirmed empirically, not just assumed) — only the scheduled/manual sync job picks up a merge.

**Image-change detection**: hash the fetched image *bytes*, not the `image_path` URL — Stash's cache-busting query param on that URL tracks `updated_at`, which changes on *any* field edit, not just a cover swap. Hashing the URL instead of the bytes was tried and reverted after it caused every unrelated performer edit to trigger a pointless re-embed.

**Voyager gotcha**: `del index[id]` (mark-deleted) can raise `RuntimeError: already deleted` even immediately after `id in index` reported it present — a tombstone-state inconsistency that surfaces after a save()/load() round-trip, not a logic bug in the calling code. `LocalPerformerIndex.remove()` treats that specific error as a no-op.

## Conventions

- **Logging:** Default level is WARNING. `logger.warning()` is user-visible; `logger.info()` is not.
- **Rate limiting:** Shared 5 req/s for Stash and StashBox APIs. StashBox calls use `Priority.LOW`.
- **Plugin logging:** Use Stash log protocol with level-prefix bytes (`\x01` + level_char + `\x02`), not plain JSON to stderr. See `stash_sense_backend.py:_log_prefix()`.
- **Local-only fields:** `favorite`, `rating`, `o_count` are Stash-local metadata — never compare against upstream StashBox values.
- **Test marking:** ML/GPU tests are marked `@pytest.mark.heavy`. CI runs `make test-ci` which excludes them. `conftest.py` mocks ML modules so heavy-marked files can be collected without GPU.
- **Background tasks:** Don't inherit shell activation. Use explicit venv python path for background processes.
- **Never auto-select in Stash's own react-select fields from plugin JS after a mutation already succeeded.** If a performer/tag/etc. was just added via a direct GraphQL mutation, Stash's own field for that entity correctly *excludes* it from its own search suggestions (it's already selected server-side) — typing its name and confirming with Enter will pick whatever else matches instead, which can be a wrong, unrelated entity (confirmed live: a name/alias collision added a completely different performer to a scene). The mutation alone is sufficient; don't also try to fake a visual update via DOM simulation. If visual feedback matters, check the entity's own Save button (`.edit-button`, filter by text "Save"; its `disabled` state is the form's dirty flag) to decide whether it's safe to `window.location.reload()` (no unsaved changes) or whether to just show a message instead (unsaved changes present, don't touch the form).
- **UI verification**: Chromium + Playwright are available locally (on the machine running the assistant, not on `<stash-host>`) for real visual verification of plugin changes (screenshot a scene/settings/operations page after a deploy) rather than assuming a UI change works. Point the browser at the deployed Stash instance's URL over the network. Exact binary/cache paths aren't fixed enough across environments to hardcode here.

## Field Name Mapping (Upstream Sync)

Stash-box uses separate fields that Stash combines into compound strings:

| Diff Engine | Stash Mutation | Notes |
|---|---|---|
| `aliases` | `alias_list` | |
| `height` | `height_cm` | Integer |
| `breast_type` | `fake_tits` | |
| `career_start_year` + `career_end_year` | `career_length` | Combined "YYYY-YYYY" |
| `cup_size` + `band_size` + `waist_size` + `hip_size` | `measurements` | Combined "38F-24-35" |

Translation: `recommendations_router.py:update_performer_fields()`

## Key Files

- `api/main.py` — App entry point, lifespan, router wiring, lazy-load setup
- `api/recommendations_router.py` — All recommendation API endpoints
- `api/recommendations_db.py` — SQLite layer (schema v9), migrations
- `api/queue_router.py` / `api/queue_manager.py` — Job queue API and execution engine
- `api/job_models.py` — `JOB_REGISTRY` and all job type definitions
- `api/base_job.py` — `BaseJob` ABC and `JobContext`
- `api/settings_router.py` — Settings and system info API
- `api/upstream_field_mapper.py` — Field mapping, parsing, 3-way diff engine
- `api/analyzers/base_upstream.py` — Base class with logic versioning
- `api/resource_manager.py` — Lazy load / idle-unload for face recognition
- `api/stash_client_unified.py` — Stash GraphQL client
- `api/stashbox_client.py` — StashBox GraphQL client
- `api/local_performer_index.py` — Local performer Voyager index (build/query/sync), see "Local Performer Database" above
- `api/jobs/local_performer_sync_job.py` — Full diff-and-embed sync job for the local performer index
- `plugin/stash-sense-recommendations.js` — Recommendations dashboard UI
- `plugin/stash-sense-settings.js` — Settings and model management UI
- `plugin/stash-sense-operations.js` — Operation queue UI
- `plugin/stash-sense.css` — All styles
- `plugin/stash_sense_backend.py` — Plugin backend proxy
- `changelog.txt` (repo root) — every version bump gets an entry here; see "Version bump" and "Changelog" under Commands above
