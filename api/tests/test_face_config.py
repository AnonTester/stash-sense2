"""Tests for face_config.py - face recognition constants."""

from face_config import (
    MAX_DISTANCE,
    NUM_FRAMES,
    START_OFFSET_PCT,
    END_OFFSET_PCT,
    MIN_FACE_SIZE,
    MIN_FACE_CONFIDENCE,
    CLUSTER_THRESHOLD,
    TOP_K,
    SPRITE_MAX_FRAMES,
)


def test_max_distance_in_valid_range():
    assert 0 < MAX_DISTANCE <= 1.0


def test_num_frames_positive_integer():
    assert isinstance(NUM_FRAMES, int)
    assert NUM_FRAMES > 0


def test_offsets_valid():
    assert 0 <= START_OFFSET_PCT < END_OFFSET_PCT <= 1.0


def test_face_detection_thresholds():
    assert MIN_FACE_SIZE > 0
    assert 0 < MIN_FACE_CONFIDENCE <= 1.0


def test_cluster_threshold_positive():
    assert 0 < CLUSTER_THRESHOLD <= 1.0


def test_top_k_positive():
    assert TOP_K > 0


def test_sprite_max_frames_covers_more_than_frame_sampling():
    # Sprite tiles are cheap (no decode/seek) relative to NUM_FRAMES'
    # extracted frames, so the sprite cap should never be the tighter
    # bound -- otherwise sprite-sheet identification would see less of
    # the scene than frame-extraction identification.
    assert SPRITE_MAX_FRAMES > 0
    assert SPRITE_MAX_FRAMES >= NUM_FRAMES
