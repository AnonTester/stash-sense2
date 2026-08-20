"""Shared face recognition configuration.

Single source of truth for parameters used by scene identification,
fingerprint generation, and face matching. Change values here to
tune all processes at once.
"""

# Frame extraction
NUM_FRAMES = 60           # Frames to sample per scene (tuned 2026-02-12 re-eval)
START_OFFSET_PCT = 0.05   # Skip first 5% (logos, intros)
END_OFFSET_PCT = 0.95     # Skip last 5% (credits, outros)

# Face detection
MIN_FACE_SIZE = 40        # Minimum face dimension in pixels
MIN_FACE_CONFIDENCE = 0.5 # Detection confidence threshold

# Matching
# MAX_DISTANCE below is the legacy (FaceNet512+ArcFace fusion) tuned value,
# carried over unchanged as a placeholder for the buffalo_l migration --
# NOT re-validated for the new single-embedding distance space. Needs its
# own re-tuning pass (reuse face-pipeline-bench's sidecar-faithful
# recognition_bench methodology) before this is treated as correct.
MAX_DISTANCE = 0.5
TOP_K = 3                 # Top matches per person
CLUSTER_THRESHOLD = 0.6   # Cosine distance threshold for face clustering

# Sprite-sheet identification. A sprite tile (~160x90px) is far smaller than
# an ffmpeg-extracted frame (e.g. 1920x1080) -- ~144x fewer pixels -- and
# costs no decode/seek, so unlike NUM_FRAMES this isn't tuned as a
# cost/accuracy tradeoff: default to processing every tile the sheet has,
# capped only to bound pathologically long videos with very large sheets.
SPRITE_MAX_FRAMES = 300
