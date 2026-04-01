"""
Tests for YoloDetector — YOLO inference, result parsing, and frame annotation.

Requires: yolo_models/rf3v1.pt and tests/images/ with at least one test image.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PerceptionProcessing.YoloDetector import YoloDetector
from PerceptionProcessing.Detection import Detection

MODEL = ROOT / "yolo_models" / "rf3v1.pt"
IMAGE = ROOT / "tests" / "images" / "guy_waving.jpg"


@pytest.fixture(scope="module")
def detector():
    return YoloDetector(model_path=str(MODEL), confidence_threshold=0.5, device="cpu")


@pytest.fixture(scope="module")
def sample_frame():
    frame = cv2.imread(str(IMAGE))
    assert frame is not None, f"Could not load test image: {IMAGE}"
    return frame


# ── Initialization ───────────────────────────────────────────────

def test_init_loads_model(detector):
    assert detector.model is not None
    assert len(detector.class_names) == 4


def test_init_stores_config(detector):
    assert detector.confidence_threshold == 0.5
    assert detector.device == "cpu"


# ── detect() ─────────────────────────────────────────────────────

def test_detect_returns_detections_and_annotated(detector, sample_frame):
    detections, annotated = detector.detect(sample_frame, return_annotated=True)
    assert isinstance(detections, list)
    assert all(isinstance(d, Detection) for d in detections)
    assert annotated is not None
    assert annotated.shape == sample_frame.shape


def test_detect_no_annotated(detector, sample_frame):
    detections, annotated = detector.detect(sample_frame, return_annotated=False)
    assert isinstance(detections, list)
    assert annotated is None


def test_detect_empty_frame(detector):
    empty = np.array([], dtype=np.uint8)
    detections, annotated = detector.detect(empty)
    assert detections == []
    assert annotated is None


def test_detect_none_frame(detector):
    detections, annotated = detector.detect(None)
    assert detections == []


def test_detect_black_frame(detector):
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, _ = detector.detect(black)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detection_fields(detector, sample_frame):
    detections, _ = detector.detect(sample_frame, return_annotated=False)
    if not detections:
        pytest.skip("No detections on test image")
    det = detections[0]
    assert isinstance(det.class_name, str)
    assert 0.0 <= det.confidence <= 1.0
    assert len(det.bbox_pixels) == 4
    assert len(det.bbox_normalized) == 4
    for v in det.bbox_normalized:
        assert 0.0 <= v <= 1.0


def test_high_confidence_threshold_reduces_detections(sample_frame):
    low = YoloDetector(str(MODEL), confidence_threshold=0.3, device="cpu")
    high = YoloDetector(str(MODEL), confidence_threshold=0.9, device="cpu")
    dets_low, _ = low.detect(sample_frame, return_annotated=False)
    dets_high, _ = high.detect(sample_frame, return_annotated=False)
    assert len(dets_high) <= len(dets_low)


# ── annotate_frame() ─────────────────────────────────────────────

def test_annotate_frame_draws_boxes(detector, sample_frame):
    det = Detection(
        class_id=0, class_name="person", confidence=0.9,
        bbox_normalized=(0.1, 0.1, 0.5, 0.5),
        bbox_pixels=(50, 50, 200, 200),
    )
    annotated = detector.annotate_frame(sample_frame.copy(), [det])
    assert annotated.shape == sample_frame.shape
    assert not np.array_equal(annotated, sample_frame)


def test_annotate_frame_empty_detections(detector, sample_frame):
    annotated = detector.annotate_frame(sample_frame.copy(), [])
    assert annotated is not None
