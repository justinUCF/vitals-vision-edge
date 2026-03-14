"""
test_yolo.py — YoloDetector unit tests

Tests YOLO model loading, inference, detection parsing, annotation,
and confidence filtering. No external services required.

Run:
    python tests/test_yolo.py
    pytest tests/test_yolo.py -v
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PerceptionProcessing.YoloDetector import YoloDetector
from PerceptionProcessing.Detection import Detection

# ── shared paths ─────────────────────────────────────────────────────────────
MODEL = ROOT / "yolo_models" / "rf3v1.pt"
IMAGE = ROOT / "tests" / "images" / "drone_testing1.jpg"
OUTPUT = ROOT / "tests" / "output"


# ── helpers ──────────────────────────────────────────────────────────────────

def _header(n, title):
    print(f"\n{'='*70}")
    print(f"  TEST {n}: {title}")
    print(f"{'='*70}")

def _ok(msg):  print(f"  PASS  {msg}")
def _skip(msg): print(f"  SKIP  {msg}")


# ── tests ────────────────────────────────────────────────────────────────────

def test_initialization():
    """1 — Model loads without error on CPU."""
    _header(1, "Model Initialization")
    detector = YoloDetector(model_path=str(MODEL), device="cpu")
    assert detector.model is not None
    assert len(detector.class_names) > 0
    _ok(f"Model loaded — {len(detector.class_names)} classes: {list(detector.class_names.values())}")
    return detector


def test_detect_from_path(detector):
    """2 — detect_from_path() returns Detection objects."""
    _header(2, "detect_from_path()")
    detections, annotated = detector.detect_from_path(str(IMAGE), return_annotated=True)
    assert len(detections) > 0,        "Expected at least one detection in test image"
    assert annotated is not None,       "Annotated image should be returned"
    assert annotated.ndim == 3,         "Annotated image should be 3-channel"
    for d in detections:
        assert isinstance(d, Detection)
        assert 0.0 <= d.confidence <= 1.0
        assert len(d.bbox_pixels) == 4
        assert len(d.bbox_normalized) == 4
    _ok(f"Detected {len(detections)} object(s)")
    for d in detections:
        _ok(f"  {d.class_name:15s}  conf={d.confidence:.2f}  bbox={d.bbox_pixels}")
    return detections, annotated


def test_detect_numpy(detector):
    """3 — detect() accepts a numpy BGR array."""
    _header(3, "detect() from numpy array")
    frame = cv2.imread(str(IMAGE))
    assert frame is not None
    detections, annotated = detector.detect(frame, return_annotated=True)
    assert annotated.shape == frame.shape, "Annotated should match input shape"
    _ok(f"numpy input accepted — {len(detections)} detection(s)")


def test_detect_no_annotated(detector):
    """4 — return_annotated=False returns None for annotated image."""
    _header(4, "return_annotated=False")
    detections, annotated = detector.detect_from_path(str(IMAGE), return_annotated=False)
    assert annotated is None
    _ok("annotated image is None as expected")


def test_high_confidence_threshold():
    """5 — Raising confidence threshold reduces detection count."""
    _header(5, "Confidence threshold filtering")
    low  = YoloDetector(model_path=str(MODEL), confidence_threshold=0.1, device="cpu")
    high = YoloDetector(model_path=str(MODEL), confidence_threshold=0.9, device="cpu")
    dets_low,  _ = low.detect_from_path(str(IMAGE),  return_annotated=False)
    dets_high, _ = high.detect_from_path(str(IMAGE), return_annotated=False)
    assert len(dets_low) >= len(dets_high), \
        "Lower threshold should yield >= detections as higher threshold"
    _ok(f"conf=0.10 → {len(dets_low)} dets   conf=0.90 → {len(dets_high)} dets")


def test_filter_detections(detector):
    """6 — filter_detections() by confidence and class."""
    _header(6, "filter_detections()")
    detections, _ = detector.detect_from_path(str(IMAGE), return_annotated=False)
    if not detections:
        _skip("no detections — skipping filter test")
        return

    first_class = detections[0].class_name
    filtered = detector.filter_detections(detections, class_filter=[first_class])
    assert all(d.class_name == first_class for d in filtered)
    _ok(f"class_filter=['{first_class}'] → {len(filtered)}/{len(detections)} detections kept")

    above = detector.filter_detections(detections, min_confidence=0.9)
    assert all(d.confidence >= 0.9 for d in above)
    _ok(f"min_confidence=0.9 → {len(above)}/{len(detections)} detections kept")


def test_bbox_normalization(detector):
    """7 — Normalized bbox coordinates are within [0, 1]."""
    _header(7, "bbox normalization")
    detections, _ = detector.detect_from_path(str(IMAGE), return_annotated=False)
    for d in detections:
        x0, y0, x1, y1 = d.bbox_normalized
        assert 0.0 <= x0 <= 1.0 and 0.0 <= y0 <= 1.0, f"Normalized coords out of range: {d.bbox_normalized}"
        assert 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0, f"Normalized coords out of range: {d.bbox_normalized}"
        assert x0 < x1 and y0 < y1,                    f"Invalid bbox ordering: {d.bbox_normalized}"
    _ok(f"All {len(detections)} bboxes have valid normalized coords")


def test_annotated_image_saved(annotated):
    """8 — Annotated image can be saved to disk."""
    _header(8, "Save annotated image")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT / "test_yolo_annotated.jpg"
    ok = cv2.imwrite(str(out_path), annotated)
    assert ok, "cv2.imwrite() returned False"
    _ok(f"Annotated image saved → {out_path}")


def test_empty_frame(detector):
    """9 — Empty frame is handled gracefully."""
    _header(9, "Empty frame handling")
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    detections, annotated = detector.detect(empty, return_annotated=False)
    assert detections == []
    assert annotated is None
    _ok("Empty frame returned [] and None without crashing")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*22 + "YOLO DETECTOR TEST SUITE" + " "*22 + "#")
    print("#"*70)
    print(f"  Model : {MODEL}")
    print(f"  Image : {IMAGE}")

    detector = test_initialization()
    detections, annotated = test_detect_from_path(detector)

    tests = [
        ("numpy input",             lambda: test_detect_numpy(detector)),
        ("no annotated image",      lambda: test_detect_no_annotated(detector)),
        ("confidence threshold",    test_high_confidence_threshold),
        ("filter_detections",       lambda: test_filter_detections(detector)),
        ("bbox normalization",      lambda: test_bbox_normalization(detector)),
        ("save annotated image",    lambda: test_annotated_image_saved(annotated)),
        ("empty frame",             lambda: test_empty_frame(detector)),
    ]

    passed, failed, errors = 0, 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((name, exc))
            print(f"  FAIL  {name}: {exc}")

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed")
    if errors:
        for name, exc in errors:
            print(f"    - {name}: {exc}")
    print("="*70)
    return failed == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
