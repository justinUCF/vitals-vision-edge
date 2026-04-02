"""
Tests for the full VITALS Vision Edge pipeline — end-to-end integration.

Runs the same pipeline as main.py but with a test image instead of a live
camera. No USB camera, flight controller, or Ollama server required.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from DataFusion.Datafusion import DataFusion
from Network.TCPSender import TCPSender
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


# ── Pipeline stage 1: YOLO detection ────────────────────────────

def test_yolo_produces_detections(engine, sample_frame):
    detections, _ = engine.process_image(
        sample_frame, drone_id=1, return_annotated=False
    )
    assert isinstance(detections, list)
    assert len(detections) > 0


# ── Pipeline stage 2: DataFusion tracking ────────────────────────

def test_fusion_builds_stable_tracks(engine, sample_frame):
    fusion = DataFusion(min_track_frames=2)
    for _ in range(3):
        dets, _ = engine.process_image(
            sample_frame, drone_id=1, return_annotated=False
        )
        fusion.process_detections(dets, drone_id=1)
    stable = fusion.get_stable_tracks()
    assert len(stable) > 0


# ── Pipeline stage 3: MCP message generation ─────────────────────

def test_mcp_messages_from_stable_tracks(engine, sample_frame):
    fusion = DataFusion(min_track_frames=2)
    for _ in range(3):
        dets, _ = engine.process_image(
            sample_frame, drone_id=1, return_annotated=False
        )
        fusion.process_detections(dets, drone_id=1)

    msgs = fusion.create_mcp_messages(uav_id="UAV_1")
    assert len(msgs) > 0
    det_msg = msgs[0]
    assert det_msg["schema"] == "mcp.v0.2"
    assert det_msg["type"] == "MCP.Detection"
    assert "payload" in det_msg


# ── Pipeline stage 4: MCP messages are JSON-serializable ─────────

def test_mcp_messages_serializable(engine, sample_frame):
    fusion = DataFusion(min_track_frames=2)
    for _ in range(3):
        dets, _ = engine.process_image(
            sample_frame, drone_id=1, return_annotated=False
        )
        fusion.process_detections(dets, drone_id=1)

    msgs = fusion.create_mcp_messages(uav_id="UAV_TEST")
    for msg in msgs:
        serialized = json.dumps(msg)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["schema"] == "mcp.v0.2"


# ── Pipeline stage 5: TCPSender accepts MCP messages ─────────────

def test_tcp_sender_accepts_mcp_messages(engine, sample_frame):
    fusion = DataFusion(min_track_frames=2)
    for _ in range(3):
        dets, _ = engine.process_image(
            sample_frame, drone_id=1, return_annotated=False
        )
        fusion.process_detections(dets, drone_id=1)

    msgs = fusion.create_mcp_messages(uav_id="UAV_1")
    sender = TCPSender("127.0.0.1", 59990, retry_interval=0.5)
    try:
        for msg in msgs:
            sender.send(msg)
        assert sender.buffer_size() >= len(msgs)
    finally:
        sender.close()


# ── Full pipeline: detection → tracking → MCP ────────────────────

def test_full_pipeline_person_detection(engine, sample_frame):
    fusion = DataFusion(min_track_frames=2)
    captioned_tracks = set()

    for _ in range(4):
        dets, _ = engine.process_image(
            sample_frame, drone_id=1, return_annotated=False
        )
        fusion.process_detections(dets, drone_id=1)

    for track in fusion.get_stable_tracks():
        if track.class_name.lower() != "person":
            continue
        if track.track_id in captioned_tracks:
            continue

        best_det = track.get_best_detection()
        best_det.enrich(caption="Test fallback caption")

        msgs = fusion.create_mcp_messages(
            tracks=[track], uav_id="UAV_1", include_captions=True
        )
        assert len(msgs) == 2
        assert msgs[0]["type"] == "MCP.Detection"
        assert msgs[1]["type"] == "MCP.Caption"
        assert msgs[1]["payload"]["caption"] == "Test fallback caption"
        captioned_tracks.add(track.track_id)

    assert len(captioned_tracks) > 0
