"""Quality scoring for scene fingerprint matches.

Scores a candidate match between a local scene and a stash-box scene
based on fingerprint overlap, type, duration agreement, and community votes.
"""

from typing import Optional

DURATION_TOLERANCE_SECONDS = 5.0
EXACT_HASH_ALGORITHMS = {"MD5", "OSHASH"}
DEFAULT_MIN_COUNT = 2
DEFAULT_MIN_PERCENTAGE = 66

# Beyond DURATION_TOLERANCE_SECONDS, how fast the penalty ramps with the
# relative gap between the local file and the stash-box scene's canonical
# (most-submitted) duration. A pure fingerprint hash match says nothing
# about whether the matched *scene* is really the same cut/release as the
# local file -- a scene can have fingerprints submitted for more than one
# edit (e.g. a short teaser alongside the full-length release), and the one
# that happens to match the local file's hash isn't necessarily the one
# most submitters agree is "the scene". CANONICAL_DURATION_MISMATCH_THRESHOLD
# is the relative-diff point past which a match is no longer treated as
# high-confidence regardless of match_count/match_percentage.
DURATION_PENALTY_SCALE = 2.0
CANONICAL_DURATION_MISMATCH_THRESHOLD = 0.10


def _canonical_remote_duration(
    all_remote_fingerprints: list[dict], fallback_duration
) -> tuple[object, int]:
    """Pick the stash-box scene's canonical duration: the duration carried by
    whichever of ITS OWN fingerprints (not just the ones that matched the
    local file) has the most community submissions. Falls back to the
    scene-level `duration` field (0 submissions) when there are no
    fingerprints to look at at all."""
    best = None
    for fp in all_remote_fingerprints or []:
        if fp.get("duration") is None:
            continue
        if best is None or fp.get("submissions", 0) > best.get("submissions", 0):
            best = fp
    if best is not None:
        return best["duration"], best.get("submissions", 0)
    return fallback_duration, 0


def score_match(
    matching_fingerprints: list[dict],
    total_local_fingerprints: int,
    local_duration: float,
    all_remote_fingerprints: Optional[list[dict]] = None,
    remote_scene_duration=None,
) -> dict:
    match_count = len(matching_fingerprints)

    if total_local_fingerprints > 0:
        match_percentage = (match_count / total_local_fingerprints) * 100
    else:
        match_percentage = 0.0

    has_exact_hash = any(
        fp["algorithm"] in EXACT_HASH_ALGORITHMS for fp in matching_fingerprints
    )

    duration_diffs = []
    for fp in matching_fingerprints:
        remote_dur = fp.get("duration")
        if remote_dur is not None and local_duration is not None:
            duration_diffs.append(abs(local_duration - remote_dur))

    if duration_diffs:
        avg_diff = sum(duration_diffs) / len(duration_diffs)
        duration_agreement = avg_diff <= DURATION_TOLERANCE_SECONDS
        duration_diff = round(avg_diff, 2)
    else:
        duration_agreement = True
        duration_diff = None

    total_submissions = sum(fp.get("submissions", 0) for fp in matching_fingerprints)

    canonical_duration, canonical_submissions = _canonical_remote_duration(
        all_remote_fingerprints or [], remote_scene_duration,
    )

    duration_mismatch = False
    duration_penalty = 0.0
    canonical_duration_diff = None
    if (
        local_duration is not None
        and canonical_duration is not None
        and local_duration > 0
        and canonical_duration > 0
    ):
        canonical_duration_diff = round(abs(local_duration - canonical_duration), 2)
        if canonical_duration_diff > DURATION_TOLERANCE_SECONDS:
            relative_diff = canonical_duration_diff / max(local_duration, canonical_duration)
            duration_penalty = min(1.0, relative_diff * DURATION_PENALTY_SCALE)
            duration_mismatch = relative_diff > CANONICAL_DURATION_MISMATCH_THRESHOLD

    if duration_penalty:
        match_percentage = match_percentage * (1 - duration_penalty)

    return {
        "match_count": match_count,
        "match_percentage": round(match_percentage, 1),
        "has_exact_hash": has_exact_hash,
        "duration_agreement": duration_agreement,
        "duration_diff": duration_diff,
        "total_submissions": total_submissions,
        "canonical_duration_remote": canonical_duration,
        "canonical_duration_submissions": canonical_submissions,
        "canonical_duration_diff": canonical_duration_diff,
        "duration_mismatch": duration_mismatch,
    }


def is_high_confidence(
    match_count: int,
    match_percentage: float,
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: int = DEFAULT_MIN_PERCENTAGE,
    duration_mismatch: bool = False,
) -> bool:
    if duration_mismatch:
        return False
    return match_count >= min_count and match_percentage >= min_percentage
