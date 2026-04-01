"""
Tests for PerceptionEngine — orchestration of YOLO + VLM.

Requires: yolo_models/rf3v1.pt and tests/images/.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PerceptionProcessing.PerceptionEngine import PerceptionEngine
from PerceptionProcessing.Detection import Detection

MODEL = ROOT / "yolo_models" / "rf3v1.pt"
IMAGE = ROOT / "tests" / "images" / "drone_testing1.jpg"


@pytest.fixture(scope="module")
def engine():
    return PerceptionEngine(
        yolo_model_path=str(MODEL),
        yolo_confidence=0.5,
        caption_threshold=0.7,
        device="cpu",
    )


@pytest.fixture(scope="module")
def sample_frame():
    frame = cv2.imread(str(IMAGE))
    assert frame is not None
    return frame


# ── Initialization ───────────────────────────────────────────────

def test_init_creates_submodules(engine):
    assert engine.yolo_detector is not None
    assert engine.captioner is not None
    assert engine.caption_threshold == 0.7


def test_init_separate_yolo_device():
    eng = PerceptionEngine(
        yolo_model_path=str(MODEL),
        device="cpu",
        yolo_device="cpu",
    )
    assert eng.yolo_detector.device == "cpu"


# ── process_image() ─────────────────────────────────────────────

def test_process_image_returns_detections(engine, sample_frame):
    detections, annotated = engine.process_image(sample_frame)
    assert isinstance(detections, list)
    assert all(isinstance(d, Detection) for d in detections)
    assert annotated is not None


def test_process_image_no_annotated(engine, sample_frame):
    detections, annotated = engine.process_image(sample_frame, return_annotated=False)
    assert annotated is None


def test_process_image_enriches_drone_id(engine, sample_frame):
    detections, _ = engine.process_image(
        sample_frame, drone_id=42, return_annotated=False
    )
    for det in detections:
        assert det.drone_id == 42


def test_process_image_enriches_location(engine, sample_frame):
    detections, _ = engine.process_image(
        sample_frame, location="28.6024,-81.2001", return_annotated=False
    )
    for det in detections:
        assert det.location == "28.6024,-81.2001"


def test_process_image_no_captions_by_default(engine, sample_frame):
    detections, _ = engine.process_image(sample_frame, return_annotated=False)
    for det in detections:
        assert det.caption is None


def test_process_image_black_frame(engine):
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    detections, _ = engine.process_image(black, return_annotated=False)
    assert detections == []


def test_to_dict_serialization(engine, sample_frame):
    detections, _ = engine.process_image(
        sample_frame, drone_id=1, return_annotated=False
    )
    if not detections:
        pytest.skip("No detections on test image")
    d = detections[0].to_dict()
    assert "class_name" in d
    assert "confidence" in d
    assert "bbox_pixels" in d
    assert "bbox_normalized" in d
