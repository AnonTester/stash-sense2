"""A tight head shot has no face for buffalo_l until it has some background around it: the local performer index
retries a cover photo with 30% edge padding, and keeps the stored box valid for the ORIGINAL photo."""
from types import SimpleNamespace

import numpy as np

from embeddings import detect_local_performer_faces, pad_image_for_bbox


def _face(x, y, w, h, rotation=0.0):
    return SimpleNamespace(bbox={"x": x, "y": y, "w": w, "h": h, "rotation_applied": rotation})


class _Generator:
    """Finds a face only when the image is larger than `min_side` (a stand-in for 'needs background')."""

    def __init__(self, min_side, face):
        self.min_side, self.face, self.calls = min_side, face, []

    def detect_faces(self, image, min_confidence=0.5):
        self.calls.append(image.shape)
        return [self.face] if min(image.shape[:2]) > self.min_side else []


def _image(side=100):
    return np.zeros((side, side, 3), dtype=np.uint8)


def test_plain_detection_is_kept_without_padding():
    gen = _Generator(min_side=0, face=_face(10, 10, 50, 50))
    faces = detect_local_performer_faces(gen, _image())
    assert len(faces) == 1 and gen.calls == [(100, 100, 3)]
    assert faces[0].bbox["x"] == 10


def test_tight_head_shot_is_found_with_padding_and_box_maps_back_to_the_photo():
    # padded canvas is 160x160 (pad 30); the face sits at (40, 35) there -> (10, 5) in the photo
    gen = _Generator(min_side=100, face=_face(40, 35, 80, 90))
    faces = detect_local_performer_faces(gen, _image(100))
    assert gen.calls == [(100, 100, 3), (160, 160, 3)]
    assert faces[0].bbox == {"x": 10, "y": 5, "w": 80, "h": 90, "rotation_applied": 0.0}


def test_box_hanging_over_the_photo_edge_is_clamped():
    gen = _Generator(min_side=100, face=_face(20, 20, 150, 150))    # padded coords: spans -10..120 in the photo
    box = detect_local_performer_faces(gen, _image(100))[0].bbox
    assert (box["x"], box["y"]) == (0, 0) and box["x"] + box["w"] <= 100 and box["y"] + box["h"] <= 100


def test_roll_corrected_padded_detection_records_the_padding_instead_of_remapping():
    gen = _Generator(min_side=100, face=_face(40, 35, 80, 90, rotation=-12.0))
    bbox = detect_local_performer_faces(gen, _image(100))[0].bbox
    assert bbox["pad_applied"] == 30 and (bbox["x"], bbox["y"]) == (40, 35)


def test_nothing_found_even_with_padding():
    gen = _Generator(min_side=10_000, face=_face(0, 0, 1, 1))
    assert detect_local_performer_faces(gen, _image(100)) == []
    assert len(gen.calls) == 2


def test_pad_image_for_bbox_rebuilds_the_padded_frame_only_when_needed():
    img = _image(100)
    assert pad_image_for_bbox(img, {"x": 1}) is img
    assert pad_image_for_bbox(img, {"pad_applied": 30}).shape == (160, 160, 3)
