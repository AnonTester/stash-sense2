#!/usr/bin/env python3
"""Backend for Stash Sense plugin.

Proxies requests to the Stash Sense sidecar API to bypass browser CSP restrictions.
"""
import json
import os
import sys
import requests


def main():
    """Handle plugin operations from Stash."""
    # Read input from stdin
    input_data = json.load(sys.stdin)

    # Get operation mode and args
    args = input_data.get("args", {})

    # A hook invocation (Performer.Create/Update/Destroy.Post) has a
    # different shape than a normal task/UI call: hookContext is present
    # and there's no mode or JS-injected sidecar_url, since Stash calls
    # the exec directly rather than going through our JS.
    if "hookContext" in args:
        handle_performer_hook(input_data)
        print(json.dumps({"output": "ok"}))
        return

    mode = args.get("mode", "")


    # Get sidecar URL from args (passed by JS which reads settings)
    sidecar_url = args.get("sidecar_url", "http://localhost:5000").rstrip("/")

    if mode == "health":
        result = health_check(sidecar_url)
    elif mode == "identify_scene":
        scene_id = args.get("scene_id")
        result = identify_scene(
            sidecar_url, scene_id,
            num_frames=args.get("max_frames") or args.get("num_frames"),
            top_k=args.get("top_k"),
            max_distance=args.get("max_distance"),
            min_face_size=args.get("min_face_size"),
            scene_performer_stashdb_ids=args.get("scene_performer_stashdb_ids", []),
            use_cache=args.get("use_cache"),
            use_sprite=args.get("use_sprite"),
            skip_frame_extraction=args.get("skip_frame_extraction"),
        )
    elif mode == "identify_image":
        image_id = args.get("image_id")
        result = identify_image(sidecar_url, image_id)
    elif mode == "identify_gallery":
        gallery_id = args.get("gallery_id")
        result = identify_gallery(sidecar_url, gallery_id)
    elif mode == "identify_frame":
        result = identify_frame(
            sidecar_url, args.get("image_base64", ""),
            top_k=args.get("top_k"),
            max_distance=args.get("max_distance"),
        )
    elif mode == "fingerprint_check_scene":
        scene_id = args.get("scene_id")
        result = sidecar_get(sidecar_url, f"/recommendations/fingerprints/scene/{scene_id}")
    elif mode == "identify_scene_progress":
        scene_id = args.get("scene_id")
        result = sidecar_get(sidecar_url, f"/identify/scene/{scene_id}/progress", timeout=10)
    elif mode == "database_info":
        result = database_info(sidecar_url)
    elif mode == "db_check_update":
        # force=True only for the "Refresh" button's own explicit request --
        # the default (routine page render) must use the sidecar's own
        # cache, not hit GitHub's unauthenticated API (60 req/hour, shared
        # across every source calling from this IP) on every load. See
        # database_health_router.py's check_database_update() docstring
        # for the live incident this caused before this default existed.
        endpoint = "/database/check-update?force=true" if args.get("force") else "/database/check-update"
        result = sidecar_get(sidecar_url, endpoint)
    elif mode == "db_update":
        result = sidecar_post(sidecar_url, "/database/update", timeout=10)
    elif mode == "db_update_status":
        result = sidecar_get(sidecar_url, "/database/update/status")
    elif mode == "local_performer_stats":
        result = sidecar_get(sidecar_url, "/recommendations/local-performers/stats", timeout=10)
    elif mode == "search_performers":
        query = args.get("query", "")
        result = sidecar_post(sidecar_url, "/stash/search-performers", {"query": query})
    elif mode == "create_performer_from_stashbox":
        result = sidecar_post(sidecar_url, "/stash/create-performer", {
            "scene_id": str(args.get("scene_id", "")),
            "image_id": str(args.get("image_id", "")),
            "endpoint": args.get("endpoint", ""),
            "stashdb_id": args.get("stashdb_id", ""),
        }, timeout=30)
    elif mode == "create_performer_from_catalogue":
        result = sidecar_post(sidecar_url, "/stash/create-performer-from-catalogue", {
            "scene_id": str(args.get("scene_id", "")),
            "image_id": str(args.get("image_id", "")),
            "source": args.get("source", ""),
            "name": args.get("name", ""),
            "country": args.get("country") or None,
            "image_url": args.get("image_url") or None,
            "catalogue_url": args.get("catalogue_url") or None,
            "profile_url": args.get("profile_url") or None,
        }, timeout=30)
    elif mode == "link_performer_stashbox":
        result = sidecar_post(sidecar_url, "/stash/link-performer", {
            "scene_id": str(args.get("scene_id", "")),
            "image_id": str(args.get("image_id", "")),
            "performer_id": str(args.get("performer_id", "")),
            "stash_ids": args.get("stash_ids", []),
            "update_metadata": args.get("update_metadata", False),
        })
    elif mode == "models_status":
        result = sidecar_get(sidecar_url, "/models/status")
    elif mode == "models_download":
        model_name = args.get("model_name", "")
        result = sidecar_post(sidecar_url, f"/models/download/{model_name}", timeout=300)
    elif mode == "models_download_all":
        result = sidecar_post(sidecar_url, "/models/download-all", timeout=300)
    elif mode == "models_progress":
        result = sidecar_get(sidecar_url, "/models/download-progress")
    elif mode == "capabilities":
        result = sidecar_get(sidecar_url, "/capabilities")
    elif mode.startswith("settings_") or mode == "system_info" or mode.startswith("logs_"):
        result = handle_settings(mode, args, sidecar_url)
        if result is None:
            result = {"error": f"Unknown settings mode: {mode}"}
    elif mode.startswith("queue_"):
        result = handle_queue(mode, args, sidecar_url)
    elif mode.startswith("rec_") or mode.startswith("fp_") or mode.startswith("user_") or mode.startswith("endpoint_"):
        result = handle_recommendations(mode, args, sidecar_url)
        if result is None:
            result = {"error": f"Unknown recommendations mode: {mode}"}
    else:
        result = {"error": f"Unknown mode: {mode}"}

    # Output result
    output = {"output": result}
    print(json.dumps(output))


def _log_prefix(level_char):
    """Build Stash log level prefix."""
    return (b'\x01' + level_char + b'\x02').decode()


def log(message):
    """Log an info message to Stash."""
    print(_log_prefix(b'i') + f"[Stash Sense] {message}\n", file=sys.stderr, flush=True)


# ==================== Performer hook (local performer index sync) ====================
#
# Keeps the local performer identification index in sync with performer
# create/update/destroy in this Stash instance. Must never block or fail
# the user's actual Stash action just because the sidecar is unreachable --
# failures are queued to a local retry-cache file and flushed opportunistically
# on a later successful hook call.

# Must match the directory name this plugin is actually installed under in
# Stash (see PLUGIN_ID's comment in stash-sense-core.js for the full story --
# this had the same v1-id bug: _get_sidecar_url() below reads
# configuration.plugins[PLUGIN_ID] to resolve the sidecar URL for a hook
# call, so while this stayed 'stash-sense' the hook silently synced
# performer changes into v1's sidecar instead of v2's whenever both were
# installed side by side).
PLUGIN_ID = "stash-sense2"
PENDING_SYNC_FILENAME = "pending_local_sync.json"
HOOK_SYNC_TIMEOUT = 3
HOOK_FLUSH_LIMIT = 20


def handle_performer_hook(input_data):
    """Handle a Performer.Create/Update/Destroy.Post hook by syncing the
    affected performer into the local identification index."""
    hook_context = input_data.get("args", {}).get("hookContext", {})
    performer_id = hook_context.get("id")
    hook_type = hook_context.get("type", "")
    if performer_id is None:
        return
    event_type = _hook_event_type(hook_type)

    server = input_data.get("server_connection", {})
    pending_path = os.path.join(server.get("PluginDir", "."), PENDING_SYNC_FILENAME)

    sidecar_url = _get_sidecar_url(server)
    if not sidecar_url:
        _append_pending_sync(pending_path, performer_id, event_type)
        log("Local performer sync hook: could not resolve sidecar URL, queued for retry")
        return

    setting = sidecar_get(
        sidecar_url, "/settings/local_performer_auto_sync_enabled", timeout=HOOK_SYNC_TIMEOUT,
    )
    if "error" in setting:
        _append_pending_sync(pending_path, performer_id, event_type)
        log(f"Local performer sync hook: sidecar unreachable, queued performer {performer_id} for retry")
        return
    if not setting.get("value"):
        return  # auto-sync disabled -- nothing to do, nothing to cache

    result = sidecar_post(
        sidecar_url, "/recommendations/local-performers/sync-one",
        {"performer_id": int(performer_id), "event_type": event_type},
        timeout=HOOK_SYNC_TIMEOUT,
    )
    if "error" in result:
        _append_pending_sync(pending_path, performer_id, event_type)
        log(f"Local performer sync hook: sync failed for performer {performer_id} ({result['error']}), queued for retry")
        return

    log(f"Local performer sync hook: performer {performer_id} -> {result.get('status')}")
    _flush_pending_sync(sidecar_url, pending_path, skip_performer_id=str(performer_id))


def _hook_event_type(hook_type):
    if "Destroy" in hook_type:
        return "destroy"
    if "Create" in hook_type:
        return "create"
    return "update"


def _get_sidecar_url(server):
    """Resolve the configured sidecar URL via Stash's own GraphQL API.
    A hook invocation has no JS-injected args (unlike normal task/UI
    calls, where our JS reads settings in the browser and passes
    sidecar_url along) -- Stash calls this exec directly with only
    hookContext + server_connection on stdin."""
    try:
        port = str(server.get("Port"))
        scheme = server.get("Scheme", "http")
        host = server.get("Host") or "localhost"
        if host == "0.0.0.0":
            host = "localhost"
        cookies = {}
        session_cookie = server.get("SessionCookie")
        if session_cookie and session_cookie.get("Value"):
            cookies["session"] = session_cookie["Value"]
        response = requests.post(
            f"{scheme}://{host}:{port}/graphql",
            json={"query": "query { configuration { plugins } }"},
            cookies=cookies,
            timeout=HOOK_SYNC_TIMEOUT,
        )
        response.raise_for_status()
        plugins = response.json().get("data", {}).get("configuration", {}).get("plugins", {})
        return (plugins.get(PLUGIN_ID) or {}).get("sidecarUrl")
    except Exception as e:
        log(f"Local performer sync hook: failed to resolve sidecar URL: {e}")
        return None


def _load_pending_sync(pending_path):
    if not os.path.exists(pending_path):
        return {}
    try:
        with open(pending_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_pending_sync(pending_path, pending):
    try:
        with open(pending_path, "w") as f:
            json.dump(pending, f)
    except OSError as e:
        log(f"Local performer sync hook: failed to write retry cache: {e}")


def _append_pending_sync(pending_path, performer_id, event_type):
    pending = _load_pending_sync(pending_path)
    pending[str(performer_id)] = event_type
    _write_pending_sync(pending_path, pending)


def _flush_pending_sync(sidecar_url, pending_path, skip_performer_id=None):
    """Best-effort flush of previously-failed syncs, piggybacking on this
    successful hook call. Capped per call so a large backlog doesn't turn
    one hook invocation into a slow bulk sync -- any remainder just stays
    cached for the next successful hook (or the manual/scheduled full
    sync task, the reliable fallback path)."""
    pending = _load_pending_sync(pending_path)
    had_skip_entry = str(skip_performer_id) in pending
    pending.pop(str(skip_performer_id), None)
    if not pending:
        # Nothing left to retry -- but if removing the current event's own
        # (now-redundant) entry emptied the cache, persist that. Otherwise
        # the stale file would linger forever, since nothing else here
        # writes it back.
        if had_skip_entry:
            _write_pending_sync(pending_path, {})
        return

    remaining = dict(pending)
    for performer_id, event_type in list(pending.items())[:HOOK_FLUSH_LIMIT]:
        result = sidecar_post(
            sidecar_url, "/recommendations/local-performers/sync-one",
            {"performer_id": int(performer_id), "event_type": event_type},
            timeout=HOOK_SYNC_TIMEOUT,
        )
        if "error" not in result:
            remaining.pop(performer_id, None)
        else:
            break  # sidecar likely down again -- stop, leave the rest cached

    _write_pending_sync(pending_path, remaining)

    flushed = len(pending) - len(remaining)
    if flushed:
        log(f"Local performer sync hook: flushed {flushed} pending sync(s) from retry cache")


def health_check(sidecar_url):
    """Check sidecar health."""
    try:
        response = requests.get(f"{sidecar_url}/health", timeout=10)
        if response.ok:
            return response.json()
        return {"error": f"Health check failed: HTTP {response.status_code}"}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Connection timed out"}
    except requests.RequestException as e:
        return {"error": f"Connection failed: {e}"}


def database_info(sidecar_url):
    """Get database info from sidecar."""
    try:
        response = requests.get(f"{sidecar_url}/database/info", timeout=10)
        if response.ok:
            return response.json()
        return {"error": f"Failed to get database info: HTTP {response.status_code}"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_scene(sidecar_url, scene_id, num_frames=None, top_k=None, max_distance=None,
                    min_face_size=None, scene_performer_stashdb_ids=None,
                    use_cache=None, use_sprite=None, skip_frame_extraction=None):
    """Identify performers in a scene. Params default to sidecar's face_config when omitted."""
    if not scene_id:
        return {"error": "No scene_id provided"}

    try:
        payload = {"scene_id": str(scene_id)}
        if num_frames is not None:
            payload["num_frames"] = int(num_frames)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if max_distance is not None:
            payload["max_distance"] = float(max_distance)
        if min_face_size is not None:
            payload["min_face_size"] = int(min_face_size)
        if scene_performer_stashdb_ids:
            payload["scene_performer_stashdb_ids"] = scene_performer_stashdb_ids
        if use_cache is not None:
            payload["use_cache"] = bool(use_cache)
        if use_sprite is not None:
            payload["use_sprite"] = bool(use_sprite)
        if skip_frame_extraction is not None:
            payload["skip_frame_extraction"] = bool(skip_frame_extraction)

        log(f"Identifying scene {scene_id}")

        response = requests.post(
            f"{sidecar_url}/identify/scene",
            json=payload,
            timeout=180,  # ffmpeg extraction can take a while
        )

        if response.ok:
            result = response.json()
            log(f"Scene {scene_id}: {result.get('faces_detected', 0)} faces, {len(result.get('persons', []))} persons")
            return result

        # Try to extract error detail from response
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"

        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out - scene may be too long or sidecar is overloaded"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_frame(sidecar_url, image_base64, top_k=None, max_distance=None):
    """Identify performers in a captured video frame or crop (base64 JPEG).

    Used by the scene page's "Identify current frame" and "Select to
    identify" options -- hits the generic /identify endpoint (no Stash
    entity involved), unlike identify_scene/identify_image/identify_gallery.
    """
    if not image_base64:
        return {"error": "No image data provided"}

    try:
        payload = {"image_base64": image_base64}
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if max_distance is not None:
            payload["max_distance"] = float(max_distance)

        log("Identifying captured frame")

        response = requests.post(
            f"{sidecar_url}/identify",
            json=payload,
            timeout=30,
        )

        if response.ok:
            result = response.json()
            log(f"Frame: {result.get('face_count', 0)} faces")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_image(sidecar_url, image_id):
    """Identify performers in a single image."""
    if not image_id:
        return {"error": "No image_id provided"}

    try:
        payload = {"image_id": str(image_id)}
        log(f"Identifying image {image_id}")

        response = requests.post(
            f"{sidecar_url}/identify/image",
            json=payload,
            timeout=30,
        )

        if response.ok:
            result = response.json()
            log(f"Image {image_id}: {result.get('face_count', 0)} faces")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_gallery(sidecar_url, gallery_id):
    """Identify performers across an entire gallery."""
    if not gallery_id:
        return {"error": "No gallery_id provided"}

    try:
        payload = {"gallery_id": str(gallery_id)}
        log(f"Identifying gallery {gallery_id}")

        response = requests.post(
            f"{sidecar_url}/identify/gallery",
            json=payload,
            timeout=300,
        )

        if response.ok:
            result = response.json()
            log(f"Gallery {gallery_id}: {result.get('images_processed', 0)} images, "
                f"{len(result.get('performers', []))} performers")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Gallery identification timed out - gallery may be too large"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


# ==================== Recommendations API Proxy ====================

def sidecar_get(sidecar_url, endpoint, timeout=30):
    """GET request to sidecar."""
    try:
        response = requests.get(f"{sidecar_url}{endpoint}", timeout=timeout)
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def sidecar_post(sidecar_url, endpoint, data=None, timeout=60):
    """POST request to sidecar."""
    try:
        response = requests.post(
            f"{sidecar_url}{endpoint}",
            json=data,
            timeout=timeout,
        )
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def sidecar_put(sidecar_url, endpoint, data=None, timeout=30):
    """PUT request to sidecar."""
    try:
        response = requests.put(
            f"{sidecar_url}{endpoint}",
            json=data,
            timeout=timeout,
        )
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def sidecar_delete(sidecar_url, endpoint, timeout=30):
    """DELETE request to sidecar."""
    try:
        response = requests.delete(f"{sidecar_url}{endpoint}", timeout=timeout)
        if response.ok:
            return response.json()
        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": error_detail}
    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


# ==================== Queue API Proxy ====================

FULL_SCAN_CURSOR = "__full__"
FORCE_FULL_SCAN_USER_JOB_TYPES = {"scene_fingerprint_match", "upstream_scene_changes"}


def handle_queue(mode, args, sidecar_url):
    """Handle queue-related proxy operations."""
    if mode == "queue_list":
        params = []
        if args.get("status"):
            params.append(f"status={args['status']}")
        if args.get("type"):
            params.append(f"type={args['type']}")
        qs = f"?{'&'.join(params)}" if params else ""
        return sidecar_get(sidecar_url, f"/queue{qs}")
    elif mode == "queue_status":
        return sidecar_get(sidecar_url, "/queue/status")
    elif mode == "queue_types":
        return sidecar_get(sidecar_url, "/queue/types")
    elif mode == "queue_submit":
        payload = {
            "type": args["type"],
            "triggered_by": args.get("triggered_by", "user"),
        }
        if "cursor" in args:
            payload["cursor"] = args.get("cursor")
        elif (
            payload["triggered_by"] == "user"
            and payload["type"] in FORCE_FULL_SCAN_USER_JOB_TYPES
        ):
            # Backward-compatible safety: user-triggered jobs in this set
            # should run full scan unless caller explicitly requested incremental.
            payload["cursor"] = FULL_SCAN_CURSOR
        return sidecar_post(sidecar_url, "/queue", payload)
    elif mode == "queue_cancel":
        return sidecar_delete(sidecar_url, f"/queue/{args['job_id']}")
    elif mode == "queue_stop":
        return sidecar_post(sidecar_url, f"/queue/{args['job_id']}/stop")
    elif mode == "queue_retry":
        return sidecar_post(sidecar_url, f"/queue/{args['job_id']}/retry")
    elif mode == "queue_clear_history":
        return sidecar_delete(sidecar_url, "/queue/history")
    elif mode == "queue_schedules":
        return sidecar_get(sidecar_url, "/queue/schedules")
    elif mode == "queue_update_schedule":
        return sidecar_put(sidecar_url, f"/queue/schedules/{args['type']}", {
            "enabled": args["enabled"],
            "interval_hours": args["interval_hours"],
        })
    return {"error": f"Unknown queue mode: {mode}"}


def rec_counts(sidecar_url):
    """Get recommendation counts."""
    return sidecar_get(sidecar_url, "/recommendations/counts")


def rec_list(sidecar_url, status=None, rec_type=None, limit=100, offset=0):
    """List recommendations."""
    params = []
    if status:
        params.append(f"status={status}")
    if rec_type:
        params.append(f"type={rec_type}")
    params.append(f"limit={limit}")
    params.append(f"offset={offset}")
    query = "?" + "&".join(params) if params else ""
    # Use a longer timeout for large fetches (4000+ recs with full details)
    timeout = 120 if limit > 500 else 30
    return sidecar_get(sidecar_url, f"/recommendations{query}", timeout=timeout)


def rec_get(sidecar_url, rec_id):
    """Get single recommendation."""
    return sidecar_get(sidecar_url, f"/recommendations/{rec_id}")


def rec_resolve(sidecar_url, rec_id, action, details=None):
    """Resolve a recommendation."""
    data = {"action": action}
    if details:
        data["details"] = details
    return sidecar_post(sidecar_url, f"/recommendations/{rec_id}/resolve", data)


def rec_dismiss(sidecar_url, rec_id, reason=None):
    """Dismiss a recommendation."""
    data = {"reason": reason} if reason else {}
    return sidecar_post(sidecar_url, f"/recommendations/{rec_id}/dismiss", data)


def rec_analysis_types(sidecar_url):
    """Get analysis types."""
    return sidecar_get(sidecar_url, "/recommendations/analysis/types")


def rec_run_analysis(sidecar_url, analysis_type, full=False):
    """Run an analysis."""
    query = "?full=true" if full else ""
    return sidecar_post(sidecar_url, f"/recommendations/analysis/{analysis_type}/run{query}")


def rec_analysis_runs(sidecar_url, analysis_type=None, limit=20):
    """Get recent analysis runs."""
    params = [f"limit={limit}"]
    if analysis_type:
        params.append(f"type={analysis_type}")
    query = "?" + "&".join(params)
    return sidecar_get(sidecar_url, f"/recommendations/analysis/runs{query}")


def rec_stash_status(sidecar_url):
    """Get Stash connection status."""
    return sidecar_get(sidecar_url, "/recommendations/stash/status")


def rec_merge_performers(sidecar_url, destination_id, source_ids):
    """Execute performer merge."""
    data = {
        "destination_id": destination_id,
        "source_ids": source_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/merge-performers", data, timeout=120)


def rec_delete_files(sidecar_url, scene_id, file_ids_to_delete, keep_file_id, all_file_ids):
    """Delete scene files."""
    data = {
        "scene_id": scene_id,
        "file_ids_to_delete": file_ids_to_delete,
        "keep_file_id": keep_file_id,
        "all_file_ids": all_file_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/delete-scene-files", data, timeout=120)


def rec_merge_scenes(sidecar_url, destination_id, source_ids):
    """Execute scene merge."""
    data = {
        "destination_id": destination_id,
        "source_ids": source_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/merge-scenes", data, timeout=120)


def rec_delete_scene(sidecar_url, scene_id, delete_file=False):
    """Delete a scene."""
    data = {
        "scene_id": scene_id,
        "delete_file": delete_file,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/delete-scene", data, timeout=60)


def rec_get_scene(sidecar_url, scene_id):
    """Get scene details for display."""
    return sidecar_get(sidecar_url, f"/recommendations/scene/{scene_id}")


def rec_merge_duplicate_scene_group(sidecar_url, source_scene_id, selected_match_scene_ids, selected_recommendation_ids, unselected_recommendation_ids):
    """Merge selected duplicate-scene matches into the source scene."""
    data = {
        "source_scene_id": source_scene_id,
        "selected_match_scene_ids": selected_match_scene_ids,
        "selected_recommendation_ids": selected_recommendation_ids,
        "unselected_recommendation_ids": unselected_recommendation_ids,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/merge-duplicate-scene-group", data, timeout=120)


def rec_delete_duplicate_scene_group(sidecar_url, source_scene_id, recommendation_ids, delete_file=False):
    """Delete the grouped duplicate-scene source scene."""
    data = {
        "source_scene_id": source_scene_id,
        "recommendation_ids": recommendation_ids,
        "delete_file": delete_file,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/delete-duplicate-scene-group", data, timeout=120)


def rec_delete_duplicate_scene_match(sidecar_url, source_scene_id, match_scene_id, recommendation_id, delete_file=False):
    """Delete one matched duplicate-scene and resolve its grouped review entry."""
    data = {
        "source_scene_id": source_scene_id,
        "match_scene_id": match_scene_id,
        "recommendation_id": recommendation_id,
        "delete_file": delete_file,
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/delete-duplicate-scene-match", data, timeout=120)


def rec_merge_source_into_duplicate_scene_match(sidecar_url, source_scene_id, keeper_match_scene_id, keeper_recommendation_id, other_matches=None):
    """Merge a grouped duplicate-scene source into one matched scene."""
    data = {
        "source_scene_id": source_scene_id,
        "keeper_match_scene_id": keeper_match_scene_id,
        "keeper_recommendation_id": keeper_recommendation_id,
        "other_matches": other_matches or [],
    }
    return sidecar_post(sidecar_url, "/recommendations/actions/merge-source-into-duplicate-scene-match", data, timeout=120)


def rec_dismiss_duplicate_scene_group(sidecar_url, recommendation_ids, reason=None):
    """Dismiss all duplicate-scene raw recommendations in a group."""
    data = {
        "recommendation_ids": recommendation_ids,
    }
    if reason:
        data["reason"] = reason
    return sidecar_post(sidecar_url, "/recommendations/actions/dismiss-duplicate-scene-group", data, timeout=60)


# ==================== Fingerprint Operations ====================

def fp_status(sidecar_url):
    """Get fingerprint status and coverage."""
    return sidecar_get(sidecar_url, "/recommendations/fingerprints/status")


def fp_generate(sidecar_url, refresh_outdated=True, num_frames=None, min_face_size=None, max_distance=None):
    """Start fingerprint generation. Params default to sidecar's face_config when omitted."""
    data = {"refresh_outdated": refresh_outdated}
    if num_frames is not None:
        data["num_frames"] = int(num_frames)
    if min_face_size is not None:
        data["min_face_size"] = int(min_face_size)
    if max_distance is not None:
        data["max_distance"] = float(max_distance)
    return sidecar_post(sidecar_url, "/recommendations/fingerprints/generate", data)


def fp_progress(sidecar_url):
    """Get fingerprint generation progress."""
    return sidecar_get(sidecar_url, "/recommendations/fingerprints/progress")


def fp_stop(sidecar_url):
    """Stop fingerprint generation."""
    return sidecar_post(sidecar_url, "/recommendations/fingerprints/stop", {})


def fp_reset(sidecar_url):
    """Back up all scene fingerprints, then mark every one for refresh.

    Used when a detection-affecting setting (e.g. detection_size) changes
    and the user chooses to regenerate existing fingerprints rather than
    only apply the new setting going forward.
    """
    return sidecar_post(sidecar_url, "/recommendations/fingerprints/reset", {})


def handle_recommendations(mode, args, sidecar_url):
    """Handle recommendations-related operations."""
    if mode == "rec_counts":
        return rec_counts(sidecar_url)

    elif mode == "rec_list":
        return rec_list(
            sidecar_url,
            status=args.get("status"),
            rec_type=args.get("type"),
            limit=int(args.get("limit", 100)),
            offset=int(args.get("offset", 0)),
        )

    elif mode == "rec_get":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return rec_get(sidecar_url, rec_id)

    elif mode == "rec_resolve":
        rec_id = args.get("rec_id")
        action = args.get("action")
        if not rec_id or not action:
            return {"error": "rec_id and action required"}
        return rec_resolve(sidecar_url, rec_id, action, args.get("details"))

    elif mode == "rec_dismiss":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return rec_dismiss(sidecar_url, rec_id, args.get("reason"))

    elif mode == "rec_batch_dismiss":
        rec_type = args.get("type")
        if not rec_type:
            return {"error": "No type provided"}
        return sidecar_post(sidecar_url, "/recommendations/actions/batch-dismiss", {
            "type": rec_type,
            "permanent": args.get("permanent", False),
        })

    elif mode == "rec_analysis_types":
        return rec_analysis_types(sidecar_url)

    elif mode == "rec_run_analysis":
        analysis_type = args.get("analysis_type")
        if not analysis_type:
            return {"error": "No analysis_type provided"}
        full_arg = args.get("full")
        full = bool(full_arg) if full_arg is not None else False
        if analysis_type in FORCE_FULL_SCAN_USER_JOB_TYPES:
            full = True
        return rec_run_analysis(
            sidecar_url,
            analysis_type,
            full=full,
        )

    elif mode == "rec_analysis_runs":
        return rec_analysis_runs(
            sidecar_url,
            analysis_type=args.get("analysis_type"),
            limit=int(args.get("limit", 20)),
        )

    elif mode == "rec_stash_status":
        return rec_stash_status(sidecar_url)

    elif mode == "rec_merge_performers":
        destination_id = args.get("destination_id")
        source_ids = args.get("source_ids", [])
        if not destination_id or not source_ids:
            return {"error": "destination_id and source_ids required"}
        return rec_merge_performers(sidecar_url, destination_id, source_ids)

    elif mode == "rec_delete_files":
        scene_id = args.get("scene_id")
        file_ids_to_delete = args.get("file_ids_to_delete", [])
        keep_file_id = args.get("keep_file_id")
        all_file_ids = args.get("all_file_ids", [])
        if not scene_id or not keep_file_id:
            return {"error": "scene_id and keep_file_id required"}
        return rec_delete_files(sidecar_url, scene_id, file_ids_to_delete, keep_file_id, all_file_ids)

    elif mode == "rec_merge_scenes":
        destination_id = args.get("destination_id")
        source_ids = args.get("source_ids", [])
        if not destination_id or not source_ids:
            return {"error": "destination_id and source_ids required"}
        return rec_merge_scenes(sidecar_url, destination_id, source_ids)

    elif mode == "rec_delete_scene":
        scene_id = args.get("scene_id")
        delete_file = args.get("delete_file", False)
        if not scene_id:
            return {"error": "scene_id required"}
        return rec_delete_scene(sidecar_url, scene_id, delete_file)

    elif mode == "rec_get_scene":
        scene_id = args.get("scene_id")
        if not scene_id:
            return {"error": "scene_id required"}
        return rec_get_scene(sidecar_url, scene_id)

    elif mode == "rec_merge_duplicate_scene_group":
        source_scene_id = args.get("source_scene_id")
        selected_match_scene_ids = args.get("selected_match_scene_ids", [])
        selected_recommendation_ids = args.get("selected_recommendation_ids", [])
        unselected_recommendation_ids = args.get("unselected_recommendation_ids", [])
        if not source_scene_id or not selected_match_scene_ids:
            return {"error": "source_scene_id and selected_match_scene_ids required"}
        return rec_merge_duplicate_scene_group(
            sidecar_url,
            source_scene_id,
            selected_match_scene_ids,
            selected_recommendation_ids,
            unselected_recommendation_ids,
        )

    elif mode == "rec_delete_duplicate_scene_group":
        source_scene_id = args.get("source_scene_id")
        recommendation_ids = args.get("recommendation_ids", [])
        delete_file = args.get("delete_file", False)
        if not source_scene_id or not recommendation_ids:
            return {"error": "source_scene_id and recommendation_ids required"}
        return rec_delete_duplicate_scene_group(
            sidecar_url,
            source_scene_id,
            recommendation_ids,
            delete_file,
        )

    elif mode == "rec_delete_duplicate_scene_match":
        source_scene_id = args.get("source_scene_id")
        match_scene_id = args.get("match_scene_id")
        recommendation_id = args.get("recommendation_id")
        delete_file = args.get("delete_file", False)
        if not source_scene_id or not match_scene_id or recommendation_id is None:
            return {"error": "source_scene_id, match_scene_id, and recommendation_id required"}
        return rec_delete_duplicate_scene_match(
            sidecar_url,
            source_scene_id,
            match_scene_id,
            recommendation_id,
            delete_file,
        )

    elif mode == "rec_merge_source_into_duplicate_scene_match":
        source_scene_id = args.get("source_scene_id")
        keeper_match_scene_id = args.get("keeper_match_scene_id")
        keeper_recommendation_id = args.get("keeper_recommendation_id")
        if not source_scene_id or not keeper_match_scene_id or keeper_recommendation_id is None:
            return {"error": "source_scene_id, keeper_match_scene_id, and keeper_recommendation_id required"}
        return rec_merge_source_into_duplicate_scene_match(
            sidecar_url,
            source_scene_id,
            keeper_match_scene_id,
            keeper_recommendation_id,
            args.get("other_matches", []),
        )

    elif mode == "rec_dismiss_duplicate_scene_group":
        recommendation_ids = args.get("recommendation_ids", [])
        if not recommendation_ids:
            return {"error": "recommendation_ids required"}
        return rec_dismiss_duplicate_scene_group(
            sidecar_url,
            recommendation_ids,
            args.get("reason"),
        )

    elif mode == "rec_update_performer":
        performer_id = args.get("performer_id")
        fields = args.get("fields", {})
        if not performer_id:
            return {"error": "performer_id required"}
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/update-performer",
            {"performer_id": performer_id, "fields": fields},
            timeout=60,
        )

    elif mode == "rec_update_tag":
        tag_id = args.get("tag_id")
        fields = args.get("fields", {})
        if not tag_id:
            return {"error": "tag_id required"}
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/update-tag",
            {"tag_id": tag_id, "fields": fields},
            timeout=60,
        )

    elif mode == "rec_update_studio":
        studio_id = args.get("studio_id")
        fields = args.get("fields", {})
        endpoint = args.get("endpoint", "")
        if not studio_id:
            return {"error": "studio_id required"}
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/update-studio",
            {"studio_id": studio_id, "fields": fields, "endpoint": endpoint},
            timeout=30,
        )

    elif mode == "rec_search_entities":
        return sidecar_post(sidecar_url, "/recommendations/actions/search-entities", {
            "entity_type": args.get("entity_type", ""),
            "query": args.get("query", ""),
            "endpoint": args.get("endpoint", ""),
        })

    elif mode == "rec_find_linked_entity":
        return sidecar_post(sidecar_url, "/recommendations/actions/find-linked-entity", {
            "entity_type": args.get("entity_type", ""),
            "endpoint": args.get("endpoint", ""),
            "stashbox_id": args.get("stashbox_id", ""),
        })

    elif mode == "rec_upstream_scene_preview":
        endpoint = args.get("endpoint", "")
        stashbox_id = args.get("stashbox_id", "")
        if not endpoint or not stashbox_id:
            return {"error": "endpoint and stashbox_id required"}
        return sidecar_post(sidecar_url, "/recommendations/actions/upstream-scene-preview", {
            "endpoint": endpoint,
            "stashbox_id": stashbox_id,
        })

    elif mode == "rec_link_entity":
        entity_type = args.get("entity_type", "")
        entity_id = args.get("entity_id", "")
        if not entity_type or not entity_id:
            return {"error": "entity_type and entity_id required"}
        return sidecar_post(sidecar_url, "/recommendations/actions/link-entity", {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "endpoint": args.get("endpoint", ""),
            "stashbox_id": args.get("stashbox_id", ""),
        })

    elif mode == "rec_create_performer":
        return sidecar_post(sidecar_url, "/recommendations/actions/create-performer", {
            "stashbox_data": args.get("stashbox_data", {}),
            "endpoint": args.get("endpoint", ""),
            "stashbox_id": args.get("stashbox_id", ""),
        })

    elif mode == "rec_create_tag":
        return sidecar_post(sidecar_url, "/recommendations/actions/create-tag", {
            "stashbox_data": args.get("stashbox_data", {}),
            "endpoint": args.get("endpoint", ""),
            "stashbox_id": args.get("stashbox_id", ""),
        })

    elif mode == "rec_create_studio":
        return sidecar_post(sidecar_url, "/recommendations/actions/create-studio", {
            "stashbox_data": args.get("stashbox_data", {}),
            "endpoint": args.get("endpoint", ""),
            "stashbox_id": args.get("stashbox_id", ""),
        })

    elif mode == "rec_update_scene":
        return sidecar_post(sidecar_url, "/recommendations/actions/update-scene", {
            "scene_id": str(args.get("scene_id", "")),
            "fields": args.get("fields", {}),
            "performer_ids": args.get("performer_ids"),
            "tag_ids": args.get("tag_ids"),
            "studio_id": args.get("studio_id"),
        })

    elif mode == "rec_dismiss_upstream":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return sidecar_post(
            sidecar_url,
            f"/recommendations/{rec_id}/dismiss-upstream",
            {"reason": args.get("reason"), "permanent": args.get("permanent", False)},
        )

    elif mode == "rec_get_field_config":
        import base64
        endpoint = args.get("endpoint", "")
        endpoint_b64 = base64.b64encode(endpoint.encode()).decode()
        return sidecar_get(sidecar_url, f"/recommendations/upstream/field-config/{endpoint_b64}")

    elif mode == "rec_set_field_config":
        import base64
        endpoint = args.get("endpoint", "")
        endpoint_b64 = base64.b64encode(endpoint.encode()).decode()
        return sidecar_post(
            sidecar_url,
            f"/recommendations/upstream/field-config/{endpoint_b64}",
            args.get("field_configs", {}),
        )

    elif mode == "rec_accept_fingerprint_match":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-fingerprint-match",
            {
                "recommendation_id": args.get("recommendation_id"),
                "scene_id": args.get("scene_id"),
                "endpoint": args.get("endpoint"),
                "stash_id": args.get("stash_id"),
            },
        )

    elif mode == "rec_accept_scene_face_matches":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-scene-face-matches",
            {
                "scene_id": str(args.get("scene_id", "")),
                "selections": args.get("selections", []),
            },
        )

    elif mode == "rec_reject_all_scene_face_matches":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/reject-all-scene-face-matches",
            {"scene_id": str(args.get("scene_id", ""))},
        )

    elif mode == "rec_dismiss_scene_face_match":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/dismiss-scene-face-match",
            {"rec_id": args.get("rec_id")},
        )

    elif mode == "rec_accept_all_fingerprint_matches":
        payload = {}
        endpoint = args.get("endpoint")
        if endpoint:
            payload["endpoint"] = endpoint
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-all-fingerprint-matches",
            payload,
        )

    elif mode == "rec_accept_all_scene_tag_only_changes":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-all-scene-tag-only-changes",
            {},
            timeout=300,
        )

    elif mode == "rec_accept_scene_tag_only_change":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-scene-tag-only-change",
            {"rec_id": args.get("rec_id")},
            timeout=300,
        )

    elif mode == "rec_accept_scene_change":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-scene-change",
            {"rec_id": args.get("rec_id")},
            timeout=300,
        )

    elif mode == "rec_scene_tag_only_stats":
        return sidecar_post(sidecar_url, "/recommendations/actions/scene-tag-only-stats", {})

    elif mode == "rec_performer_url_only_stats":
        return sidecar_post(sidecar_url, "/recommendations/actions/performer-url-only-stats", {})

    elif mode == "rec_fingerprint_match_stats":
        return sidecar_post(sidecar_url, "/recommendations/actions/fingerprint-match-stats", {})

    elif mode == "rec_bulk_accept_stats":
        return sidecar_post(sidecar_url, "/recommendations/actions/bulk-accept-stats", {"type": args.get("type", "")})

    elif mode == "rec_accept_all_performer_url_only_changes":
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/accept-all-performer-url-only-changes",
            {},
            timeout=300,
        )

    elif mode == "user_get_all_settings":
        return sidecar_get(sidecar_url, "/recommendations/settings")

    elif mode == "user_get_setting":
        key = args.get("key", "")
        return sidecar_get(sidecar_url, f"/recommendations/settings/{key}")

    elif mode == "user_set_setting":
        key = args.get("key", "")
        value = args.get("value")
        return sidecar_post(
            sidecar_url,
            f"/recommendations/settings/{key}",
            {"value": value},
        )

    elif mode == "endpoint_priorities_get":
        return sidecar_get(sidecar_url, "/settings/endpoint-priorities")

    elif mode == "endpoint_priorities_set":
        endpoints = args.get("endpoints", [])
        return sidecar_post(sidecar_url, "/settings/endpoint-priorities", {"endpoints": endpoints})

    elif mode == "endpoint_disable":
        endpoint = args.get("endpoint")
        if not endpoint:
            return {"error": "No endpoint provided"}
        return sidecar_post(sidecar_url, "/settings/endpoint-disable", {
            "endpoint": endpoint,
            "clear_recommendations": args.get("clear_recommendations", False),
        })

    elif mode == "endpoint_enable":
        endpoint = args.get("endpoint")
        if not endpoint:
            return {"error": "No endpoint provided"}
        return sidecar_post(sidecar_url, "/settings/endpoint-enable", {"endpoint": endpoint})

    # Fingerprint operations
    elif mode == "fp_status":
        return fp_status(sidecar_url)

    elif mode == "fp_generate":
        return fp_generate(
            sidecar_url,
            refresh_outdated=args.get("refresh_outdated", True),
            num_frames=args.get("num_frames"),
            min_face_size=args.get("min_face_size"),
            max_distance=args.get("max_distance"),
        )

    elif mode == "fp_progress":
        return fp_progress(sidecar_url)

    elif mode == "fp_stop":
        return fp_stop(sidecar_url)

    elif mode == "fp_reset":
        return fp_reset(sidecar_url)

    return None


# ==================== Settings API Proxy ====================

def handle_settings(mode, args, sidecar_url):
    """Handle settings-related operations."""
    if mode == "settings_get_all":
        return sidecar_get(sidecar_url, "/settings")

    elif mode == "settings_get":
        key = args.get("key", "")
        return sidecar_get(sidecar_url, f"/settings/{key}")

    elif mode == "settings_update":
        key = args.get("key", "")
        value = args.get("value")
        body = {"value": value}
        plugin_version = args.get("plugin_version")
        if plugin_version:
            body["plugin_version"] = plugin_version
        return sidecar_put(sidecar_url, f"/settings/{key}", body)

    elif mode == "settings_update_bulk":
        settings = args.get("settings", {})
        return sidecar_put(sidecar_url, "/settings", {"settings": settings})

    elif mode == "settings_reset":
        key = args.get("key", "")
        return sidecar_delete(sidecar_url, f"/settings/{key}")

    elif mode == "system_info":
        return sidecar_get(sidecar_url, "/system/info")

    elif mode == "logs_list":
        return sidecar_get(sidecar_url, "/settings/logs")

    elif mode == "logs_download":
        filename = args.get("filename", "")
        return sidecar_get(sidecar_url, f"/settings/logs/download/{filename}")

    elif mode == "logs_delete":
        filename = args.get("filename", "")
        return sidecar_delete(sidecar_url, f"/settings/logs/{filename}")

    elif mode == "logs_delete_all":
        return sidecar_delete(sidecar_url, "/settings/logs")

    elif mode == "logs_download_all":
        return sidecar_get(sidecar_url, "/settings/logs/download-all")

    return None


if __name__ == "__main__":
    main()
