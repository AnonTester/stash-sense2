"""
Signal scoring functions for multi-signal performer identification.

Provides adjustment multipliers based on body proportion and tattoo signals
to re-rank face recognition candidates.
"""

from typing import Optional

from body_proportions import BodyProportions
from tattoo_detector import TattooResult


def body_ratio_penalty(
    query_ratios: Optional[BodyProportions],
    candidate_ratios: Optional[BodyProportions],
) -> float:
    """
    Compute adjustment multiplier based on body proportion comparison.

    Compares the shoulder_hip_ratio (most discriminating metric) between
    query and candidate proportions.

    Args:
        query_ratios: Body proportions from the query image (may be None)
        candidate_ratios: Body proportions from the candidate (may be None)

    Returns:
        Adjustment multiplier:
        - 1.0 if either input is None (no penalty)
        - 1.0 if diff <= 0.12 (compatible)
        - 0.85 if diff > 0.12 (slight mismatch)
        - 0.6 if diff > 0.2 (moderate mismatch)
        - 0.3 if diff > 0.35 (severe mismatch)
    """
    if query_ratios is None or candidate_ratios is None:
        return 1.0

    diff = abs(query_ratios.shoulder_hip_ratio - candidate_ratios.shoulder_hip_ratio)

    if diff > 0.35:
        return 0.3
    elif diff > 0.2:
        return 0.6
    elif diff > 0.12:
        return 0.85
    else:
        return 1.0


def tattoo_adjustment(
    query_result: Optional[TattooResult],
    candidate_id: str,
    tattoo_scores: Optional[dict[str, float]] = None,
    has_tattoo_embeddings: bool = False,
) -> float:
    """
    Compute adjustment multiplier based on tattoo embedding similarity.

    Deliberately a corroborating/tie-breaking signal, not a strong
    identifier on its own -- a real matching-accuracy pass (100 tattoo-
    tagged scenes vs 100 non-tattoo control scenes, reference embeddings
    built from performers' own scene screenshots) found the older, wider
    multiplier range (0.7-1.5x) net-negative: it moved very few scenes,
    and every move it made was neutral-to-harmful, including at least one
    case where it flipped an already-correct face match into a wrong one.
    Published research on this exact detector+embedder combination reports
    only ~0.52 F-score for trusting a single best cosine match. Narrowed
    the whole range to roughly +-8% so it can nudge between near-tied face
    candidates without ever overturning a clear face-based leader, and
    softened the "candidate has no reference embeddings" case from an
    outright penalty to a mild one -- absence of a reference embedding
    usually just means nobody's built one yet, not evidence the candidate
    lacks a tattoo, so it shouldn't be scored as if it were.

    Uses visual similarity scores from TattooMatcher (kNN on CLIP ViT-B/32
    embeddings) instead of binary has/doesn't-have presence.

    Args:
        query_result: Tattoo detection result from the query image (may be None)
        candidate_id: Universal ID of the candidate performer
        tattoo_scores: Dict of universal_id -> best similarity score from TattooMatcher
        has_tattoo_embeddings: Whether the candidate has any tattoo embeddings in the index

    Returns:
        Adjustment multiplier:
        - 1.0 if query_result is None or no tattoos detected (neutral)
        - 1.05-1.08 if high tattoo similarity (>0.7) between query and candidate
        - 1.03 if moderate tattoo similarity (>0.5)
        - 0.97 if query has tattoos but candidate has no tattoo embeddings (mild, not a penalty for absent data)
        - 0.98 if query has no tattoos but candidate has many tattoo embeddings
        - 1.0 otherwise (neutral)
    """
    if query_result is None:
        return 1.0

    query_has_tattoos = query_result.has_tattoos

    # No tattoos in query image
    if not query_has_tattoos:
        if has_tattoo_embeddings:
            return 0.98  # Slight penalty: candidate has tattoos, query doesn't
        return 1.0

    # Query has tattoos — check embedding similarity scores
    if tattoo_scores:
        score = tattoo_scores.get(candidate_id, 0.0)
        if score > 0.7:
            # High similarity — modest boost (scale linearly 1.05-1.08)
            return 1.05 + (score - 0.7) * (0.03 / 0.3)
        elif score > 0.5:
            return 1.03  # Moderate similarity — small nudge

    # Query has tattoos but candidate has no tattoo embeddings at all --
    # mild down-weight, not a penalty: this usually just reflects sparse
    # reference coverage, not positive evidence the candidate lacks a tattoo.
    if not has_tattoo_embeddings:
        return 0.97

    # Query has tattoos, candidate has embeddings, but low/no similarity
    return 1.0
