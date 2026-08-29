"""Benchmark: does sizing InsightFace's detector input to a sprite tile's
*actual* dimensions (instead of the pipeline's fixed 640x640) change
detection/embedding speed or matching accuracy?

Background: Refresh Outdated (sidecar 0.21.3) now runs sprite-tile
detection for every scene once (SPRITE_MAX_FRAMES=300 tiles/scene, vs
NUM_FRAMES=60 video frames/scene -- see face_config.py and
identification_router.py's _process_sprite_frames), and face_config.py's
own SPRITE_MAX_FRAMES comment assumes this is cheap because a sprite tile
(~160x90px) has "~144x fewer pixels" than a full video frame. That
assumption doesn't hold given how the pipeline actually calls the
detector: embeddings.py's FaceEmbeddingGenerator.face_analyzer always
runs at a fixed det_size (default 640x640, see face_analyzer's own
settings-driven prepare() call) -- InsightFace's SCRFD detector resizes
and letterboxes *every* input onto that fixed square canvas before its
forward pass, so a 160x90 tile currently costs the same detector compute
as a full frame, regardless of its tiny native resolution.

This script tests the fix suggested in review: give the detector a
det_size matched to each tile's actual (VTT-reported) dimensions instead,
rounded up to the nearest multiple of 32 (SCRFD's detection head uses
strides up to 32, so det_size must be a multiple of that). It measures,
over a real sample of sprite tiles pulled from live Stash scenes:

  1. Wall-clock detect_faces() time -- fixed 640x640 vs actual-dims.
  2. Whether the resulting embeddings still match the same performers --
     each tile's detected face is matched against the real production
     performer database (recognizer.recognize_face_v2), and the top-1
     match is checked against that scene's actual Stash-assigned
     performers (ground truth).

InsightFace's prepare() only updates the stored input-size used for
resize/letterbox -- it does not reload the ONNX session/weights -- so
switching det_size between groups of tiles mid-run is cheap and doesn't
require reloading buffalo_l.

Must run wherever the real GPU, real Stash instance, and real performer
database live (homeserver's /opt/stash-sense2 deployment) -- this reuses
the exact production FaceRecognizer/buffalo_l pipeline and a live Stash
GraphQL connection, not a synthetic/offline dataset. Do not run against
a dev machine's local Stash unless you specifically want to benchmark
that box.

That host previously had a history of ROCm crashes from an earlier
evaluation (/opt/face-pipeline-bench's core dumps), since resolved (root
cause was mixing video-decode and compute GPU tasks -- unstable on this
iGPU -- plus an incorrect ROCm override version); this script still
writes results to CSV incrementally, one row per completed tile, and
isolates each tile's detect+match in its own try/except, purely as cheap
insurance against losing a long run to something unrelated (e.g. an OOM).

Sampling target is scene count, not tile count: --target-scenes distinct
scenes (default 500) each contribute up to --tiles-per-scene tiles (default
face_config.SPRITE_MAX_FRAMES=300, matching what production's Refresh
Outdated actually pulls per scene) from their own sprite sheet -- a
scene's real sheet (Stash generates 10-500 tiles depending on duration)
contributes fewer if it has fewer. Total tiles therefore scales with the
real per-scene sprite sizes across 500 real scenes, not a fixed count.

Usage (from api/, with the sidecar's own venv/env active so STASH_URL/
STASH_API_KEY/DATA_DIR resolve to the real deployment):

    python -m benchmark.sprite_detsize_benchmark --target-scenes 500

Start with a small --target-scenes (e.g. 10) first to confirm connectivity
and that nothing crashes before committing to a full 500-scene run.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import csv
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

import face_config
from config import StashConfig, DatabaseConfig
from matching import MatchingConfig
from recognizer import FaceRecognizer
from sprite_parser import fetch_sprite_from_stash

logger = logging.getLogger(__name__)

STASHDB_ENDPOINT = "https://stashdb.org/graphql"
DET_STRIDE = 32  # SCRFD's detection head uses strides up to 32 -- det_size must be a multiple of this
FIXED_DET_SIZE = (640, 640)  # current production default (embeddings.py face_analyzer.prepare)


def _stashdb_id(stash_ids: Optional[list[dict]]) -> Optional[str]:
    for sid in stash_ids or []:
        if sid.get("endpoint") == STASHDB_ENDPOINT:
            return sid.get("stash_id")
    return None


def _round_up_stride(value: int, stride: int = DET_STRIDE) -> int:
    return max(stride, ((value + stride - 1) // stride) * stride)


@dataclass
class TileSample:
    scene_id: str
    scene_title: str
    tile_index: int
    width: int
    height: int
    image: np.ndarray
    expected_performers: frozenset[str]  # StashDB ids Stash has assigned to this scene


@dataclass
class TileOutcome:
    pass_name: str          # "fixed_640" or "actual_dims"
    scene_id: str
    tile_index: int
    native_width: int
    native_height: int
    det_size: str            # "640x640" or "<w>x<h>" actually used for this call
    detect_ms: float
    match_ms: float
    face_found: bool
    top1_stashdb_id: Optional[str]
    top1_distance: Optional[float]
    correct: bool             # top-1 match is one of the scene's assigned performers
    false_positive: bool      # a confident match was returned but it's wrong
    error: Optional[str] = None


def _build_scene_query(page: int, per_page: int) -> str:
    return f"""
    query {{
        findScenes(filter: {{ page: {page}, per_page: {per_page}, sort: "random" }}) {{
            count
            scenes {{
                id
                title
                files {{ width height }}
                paths {{ sprite vtt }}
                performers {{ name stash_ids {{ endpoint stash_id }} }}
            }}
        }}
    }}
    """


async def _stash_query(base_url: str, api_key: str, query: str) -> dict:
    """Minimal async GraphQL POST -- avoids stash_client.py's sync
    `requests` dependency, which isn't installed in the Docker image
    (only in the full dev requirements.txt); httpx already is."""
    headers = {"ApiKey": api_key, "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{base_url}/graphql", json={"query": query}, headers=headers)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise RuntimeError(f"GraphQL errors: {result['errors']}")
        return result["data"]


async def collect_tiles(
    stash_url: str,
    api_key: str,
    target_scenes: int,
    tiles_per_scene: int,
    max_pages: int,
) -> list[TileSample]:
    """Pulls sprite tiles from real Stash scenes until target_scenes
    distinct scenes have contributed at least one usable tile, taking only
    up to tiles_per_scene tiles from each -- breadth across many scenes
    (and therefore many resolutions/aspect ratios/performers) is the goal,
    not depth within a handful of them."""
    samples: list[TileSample] = []
    scenes_with_samples: set[str] = set()
    seen_scene_ids: set[str] = set()
    page = 1
    per_page = 25

    while len(scenes_with_samples) < target_scenes and page <= max_pages:
        data = await _stash_query(stash_url, api_key, _build_scene_query(page, per_page))
        scene_rows = (data.get("findScenes") or {}).get("scenes", [])
        if not scene_rows:
            break

        for scene in scene_rows:
            if len(scenes_with_samples) >= target_scenes:
                break
            scene_id = scene["id"]
            if scene_id in seen_scene_ids:
                continue
            seen_scene_ids.add(scene_id)

            paths = scene.get("paths") or {}
            sprite_url, vtt_url = paths.get("sprite"), paths.get("vtt")
            if not sprite_url or not vtt_url:
                continue

            expected = frozenset(
                sid for p in (scene.get("performers") or [])
                if (sid := _stashdb_id(p.get("stash_ids")))
            )
            if not expected:
                continue  # no usable ground truth for this scene

            try:
                frames = await fetch_sprite_from_stash(
                    sprite_url, vtt_url, api_key, max_frames=tiles_per_scene,
                )
            except Exception as exc:
                logger.warning("Sprite fetch failed for scene %s: %r", scene_id, exc)
                continue
            if not frames:
                continue

            title = scene.get("title") or f"Scene {scene_id}"
            for sf in frames[:tiles_per_scene]:
                h, w = sf.image.shape[0], sf.image.shape[1]
                samples.append(TileSample(
                    scene_id=scene_id, scene_title=title, tile_index=sf.index,
                    width=w, height=h, image=sf.image, expected_performers=expected,
                ))
            scenes_with_samples.add(scene_id)

        page += 1

    return samples


def _run_pass(
    recognizer: FaceRecognizer,
    samples: list[TileSample],
    pass_name: str,
    match_config: MatchingConfig,
    ctx_id: int,
    fixed_det_size: Optional[tuple[int, int]],
    csv_writer: csv.DictWriter,
    csv_file,
) -> list[TileOutcome]:
    generator = recognizer.generator
    analyzer = generator.face_analyzer  # forces lazy load on first use

    if fixed_det_size is not None:
        groups: dict[tuple[int, int], list[TileSample]] = {fixed_det_size: samples}
    else:
        groups = {}
        for s in samples:
            key = (_round_up_stride(s.width), _round_up_stride(s.height))
            groups.setdefault(key, []).append(s)

    outcomes: list[TileOutcome] = []
    for det_size, group in groups.items():
        analyzer.prepare(ctx_id=ctx_id, det_size=det_size)
        det_label = f"{det_size[0]}x{det_size[1]}"
        logger.info("[%s] det_size=%s -- %d tile(s)", pass_name, det_label, len(group))

        for s in group:
            outcome = TileOutcome(
                pass_name=pass_name, scene_id=s.scene_id, tile_index=s.tile_index,
                native_width=s.width, native_height=s.height, det_size=det_label,
                detect_ms=0.0, match_ms=0.0, face_found=False,
                top1_stashdb_id=None, top1_distance=None,
                correct=False, false_positive=False,
            )
            try:
                t0 = time.perf_counter()
                faces = generator.detect_faces(s.image, min_confidence=face_config.MIN_FACE_CONFIDENCE)
                outcome.detect_ms = (time.perf_counter() - t0) * 1000

                faces = [
                    f for f in faces
                    if f.bbox["w"] >= face_config.MIN_FACE_SIZE and f.bbox["h"] >= face_config.MIN_FACE_SIZE
                ]
                outcome.face_found = bool(faces)

                if faces:
                    face = max(faces, key=lambda f: f.bbox["w"] * f.bbox["h"])
                    t1 = time.perf_counter()
                    # Unpack defensively -- recognize_face_v2's return arity has
                    # drifted between checkouts (2-tuple vs 3-tuple with a raw
                    # embedding appended); only `matches` (first element) is used here.
                    matches = recognizer.recognize_face_v2(face, match_config, image=s.image)[0]
                    outcome.match_ms = (time.perf_counter() - t1) * 1000
                    if matches:
                        top1 = matches[0]
                        outcome.top1_stashdb_id = top1.stashdb_id
                        outcome.top1_distance = top1.distance
                        if top1.stashdb_id in s.expected_performers:
                            outcome.correct = True
                        else:
                            outcome.false_positive = True
            except Exception as exc:  # noqa: BLE001 -- keep going, this host is known to be crash-prone
                logger.warning(
                    "Tile failed (pass=%s scene=%s tile=%d): %r",
                    pass_name, s.scene_id, s.tile_index, exc,
                )
                outcome.error = str(exc)

            outcomes.append(outcome)
            csv_writer.writerow(outcome.__dict__)
            csv_file.flush()

    return outcomes


def _summarize(outcomes: list[TileOutcome]) -> dict:
    ok = [o for o in outcomes if o.error is None]
    times = [o.detect_ms for o in ok]
    with_face = [o for o in ok if o.face_found]
    correct = [o for o in with_face if o.correct]
    false_pos = [o for o in with_face if o.false_positive]

    return {
        "tiles": len(outcomes),
        "errors": len(outcomes) - len(ok),
        "faces_found": len(with_face),
        "correct_top1": len(correct),
        "false_positive_top1": len(false_pos),
        "accuracy_of_detected": (len(correct) / len(with_face)) if with_face else None,
        "detect_ms_total": sum(times),
        "detect_ms_mean": statistics.mean(times) if times else None,
        "detect_ms_median": statistics.median(times) if times else None,
        "detect_ms_p95": (statistics.quantiles(times, n=20)[18] if len(times) >= 20 else None),
    }


def _print_summary(label: str, summary: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  Tiles processed:      {summary['tiles']} ({summary['errors']} errored)")
    print(f"  Faces found:          {summary['faces_found']}")
    print(f"  Correct top-1:        {summary['correct_top1']}")
    print(f"  Wrong top-1:          {summary['false_positive_top1']}")
    acc = summary["accuracy_of_detected"]
    print(f"  Accuracy (of faces found): {acc:.1%}" if acc is not None else "  Accuracy: n/a (no faces found)")
    print(f"  detect_faces() total: {summary['detect_ms_total']:.0f} ms")
    if summary["detect_ms_mean"] is not None:
        print(f"  detect_faces() mean:  {summary['detect_ms_mean']:.2f} ms/tile")
        print(f"  detect_faces() median:{summary['detect_ms_median']:.2f} ms/tile")
    if summary["detect_ms_p95"] is not None:
        print(f"  detect_faces() p95:   {summary['detect_ms_p95']:.2f} ms/tile")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--target-scenes", type=int, default=500,
        help="Number of distinct scenes to sample from -- breadth across scenes, not tile count, "
             "is the sampling target.",
    )
    ap.add_argument(
        "--tiles-per-scene", type=int, default=face_config.SPRITE_MAX_FRAMES,
        help="Max tiles taken from each sampled scene's own sprite sheet -- defaults to "
             "face_config.SPRITE_MAX_FRAMES (300) to match what production's Refresh Outdated "
             "actually processes per scene; a scene's real sheet (Stash generates 10-500 tiles "
             "depending on duration) contributes fewer if it has fewer.",
    )
    ap.add_argument("--max-pages", type=int, default=200, help="Safety cap on Stash pagination while collecting scenes.")
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark_results/sprite_detsize"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    stash_config = StashConfig.from_env()
    data_dir = os.environ.get("DATA_DIR", "./data")
    db_config = DatabaseConfig(data_dir=Path(data_dir))

    print("Loading face recognition models (buffalo_l)...")
    recognizer = FaceRecognizer(db_config)
    device = recognizer.generator.device
    ctx_id = 0 if device == "gpu" else -1
    print(f"Resolved device: {device} (ctx_id={ctx_id})")

    print(f"Collecting tiles from up to {args.target_scenes} distinct scenes with assigned performers...")
    samples = await collect_tiles(
        stash_config.url, stash_config.api_key, args.target_scenes, args.tiles_per_scene, args.max_pages,
    )
    scene_count = len({s.scene_id for s in samples})
    print(f"Collected {len(samples)} tiles from {scene_count} scenes")
    if not samples:
        print("No usable scenes found (need sprite+vtt paths and >=1 assigned performer with a StashDB id).")
        return 1

    dims = sorted({(s.width, s.height) for s in samples})
    print(f"Distinct native tile dimensions seen: {dims}")

    match_config = MatchingConfig(query_k=100, max_results=10, max_distance=face_config.MAX_DISTANCE)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"tile_results_{datetime.now():%Y%m%d_%H%M%S}.csv"
    fieldnames = list(TileOutcome.__dataclass_fields__.keys())

    # Baseline (current production 640x640) and actual-per-tile-dims are the
    # two the review question was about; 480/320 are fixed (no per-group
    # prepare() switch, so no repeated warm-up cost either) intermediate
    # sizes to check whether most of the win is just "smaller canvas" rather
    # than "matches the tile exactly" -- i.e. whether det_size needs to track
    # per-scene dimensions at all, or a single smaller fixed size is enough.
    pass_specs: list[tuple[str, Optional[tuple[int, int]]]] = [
        ("fixed_640", (640, 640)),
        ("fixed_480", (480, 480)),
        ("fixed_320", (320, 320)),
        ("actual_dims", None),
    ]

    summaries: dict[str, dict] = {}
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for label, det_size in pass_specs:
            desc = f"fixed {det_size[0]}x{det_size[1]}" if det_size else "actual (per-tile) dimensions"
            print(f"\n=== Pass: {label} -- det_size={desc} ===")
            outcomes = _run_pass(recognizer, samples, label, match_config, ctx_id, det_size, writer, csv_file)
            summaries[label] = _summarize(outcomes)

    # Leave the shared analyzer back on the production default so nothing
    # else touching this same process is left pointing at a benchmark det_size.
    recognizer.generator.face_analyzer.prepare(ctx_id=ctx_id, det_size=FIXED_DET_SIZE)

    for label, _ in pass_specs:
        _print_summary(label, summaries[label])

    baseline = summaries["fixed_640"]
    print("\n=== Relative to fixed_640 baseline ===")
    for label, _ in pass_specs[1:]:
        s = summaries[label]
        if baseline["detect_ms_total"] and s["detect_ms_total"] is not None:
            speedup = baseline["detect_ms_total"] / s["detect_ms_total"] if s["detect_ms_total"] else float("inf")
            print(f"{label}: {speedup:.2f}x total detect_faces() speed", end="")
        acc_b, acc_base = s["accuracy_of_detected"], baseline["accuracy_of_detected"]
        if acc_b is not None and acc_base is not None:
            print(f", accuracy delta {(acc_b - acc_base):+.1%}")
        else:
            print()

    summary_path = args.out_dir / f"summary_{datetime.now():%Y%m%d_%H%M%S}.json"
    summary_path.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "tiles_collected": len(samples),
        "scenes": scene_count,
        "native_dims_seen": dims,
        "csv_path": str(csv_path),
        **summaries,
    }, indent=2))
    print(f"\nPer-tile CSV: {csv_path}")
    print(f"Summary JSON: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
