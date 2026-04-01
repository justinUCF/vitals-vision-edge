"""
Tests for DataFusion — temporal tracking, IoU association, EMA confidence,
track lifecycle, and MCP message generation.

No external services required.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from DataFusion.Datafusion import DataFusion, TrackedDetection
from PerceptionProcessing.Detection import Detection


def _make_detection(class_name="person", confidence=0.8, bbox=(100, 100, 200, 200),
                    location=None, caption=None):
    x1, y1, x2, y2 = bbox
    w, h = 640, 480
    return Detection(
        class_id=0 if class_name == "person" else 1,
        class_name=class_name,
        confidence=confidence,
        bbox_normalized=(x1 / w, y1 / h, x2 / w, y2 / h),
        bbox_pixels=bbox,
        location=location,
        caption=caption,
    )


# ── TrackedDetection ─────────────────────────────────────────────

class TestTrackedDetection:

    def test_init_creates_track(self):
        det = _make_detection()
        track = TrackedDetection(det, frame_number=1, timestamp="2026-01-01T00:00:00Z")
        assert track.track_id.startswith("track_")
        assert track.frame_count == 1
        assert track.class_name == "person"
        assert track.confidence == det.confidence

    def test_iou_identical_boxes(self):
        det = _make_detection()
        track = TrackedDetection(det, 1, "t")
        assert track.iou((100, 100, 200, 200), (100, 100, 200, 200)) == 1.0

    def test_iou_no_overlap(self):
        det = _make_detection()
        track = TrackedDetection(det, 1, "t")
        assert track.iou((0, 0, 50, 50), (200, 200, 300, 300)) == 0.0

    def test_iou_partial_overlap(self):
        det = _make_detection()
        track = TrackedDetection(det, 1, "t")
        score = track.iou((0, 0, 100, 100), (50, 50, 150, 150))
        assert 0.0 < score < 1.0

    def test_update_matching_detection(self):
        det1 = _make_detection(bbox=(100, 100, 200, 200))
        track = TrackedDetection(det1, 1, "t1")
        det2 = _make_detection(bbox=(110, 110, 210, 210))
        assert track.update(det2, 2, "t2") is True
        assert track.frame_count == 2

    def test_update_rejects_different_class(self):
        det1 = _make_detection(class_name="person")
        track = TrackedDetection(det1, 1, "t1")
        det2 = _make_detection(class_name="car", bbox=(100, 100, 200, 200))
        assert track.update(det2, 2, "t2") is False

    def test_update_rejects_non_overlapping(self):
        det1 = _make_detection(bbox=(0, 0, 50, 50))
        track = TrackedDetection(det1, 1, "t1")
        det2 = _make_detection(bbox=(400, 400, 500, 500))
        assert track.update(det2, 2, "t2") is False

    def test_ema_confidence_smoothing(self):
        det1 = _make_detection(confidence=0.5)
        track = TrackedDetection(det1, 1, "t1")
        det2 = _make_detection(confidence=1.0, bbox=(110, 110, 210, 210))
        track.update(det2, 2, "t2")
        assert abs(track.confidence - 0.65) < 0.01

    def test_get_best_detection(self):
        det1 = _make_detection(confidence=0.5)
        track = TrackedDetection(det1, 1, "t1")
        det2 = _make_detection(confidence=0.95, bbox=(110, 110, 210, 210))
        track.update(det2, 2, "t2")
        best = track.get_best_detection()
        assert best.confidence == 0.95

    def test_is_stable(self):
        det = _make_detection()
        track = TrackedDetection(det, 1, "t")
        assert track.is_stable(min_frames=3) is False
        for i in range(2, 5):
            d = _make_detection(bbox=(100 + i, 100 + i, 200 + i, 200 + i))
            track.update(d, i, f"t{i}")
        assert track.is_stable(min_frames=3) is True

    def test_get_average_location_none(self):
        det = _make_detection()
        track = TrackedDetection(det, 1, "t")
        assert track.get_average_location() is None

    def test_get_average_location(self):
        det = _make_detection(location="28.0,-81.0")
        track = TrackedDetection(det, 1, "t")
        det2 = _make_detection(location="29.0,-82.0", bbox=(110, 110, 210, 210))
        track.update(det2, 2, "t2")
        avg = track.get_average_location()
        assert avg is not None
        lat, lon = avg.split(",")
        assert abs(float(lat) - 28.5) < 0.01
        assert abs(float(lon) - (-81.5)) < 0.01


# ── DataFusion ───────────────────────────────────────────────────

class TestDataFusion:

    def test_init(self):
        fusion = DataFusion()
        assert fusion.frame_number == 0
        assert len(fusion.active_tracks) == 0

    def test_process_creates_tracks(self):
        fusion = DataFusion()
        dets = [_make_detection(bbox=(100, 100, 200, 200))]
        tracks = fusion.process_detections(dets)
        assert len(tracks) == 1
        assert fusion.frame_number == 1

    def test_process_associates_to_existing_track(self):
        fusion = DataFusion()
        fusion.process_detections([_make_detection(bbox=(100, 100, 200, 200))])
        fusion.process_detections([_make_detection(bbox=(110, 110, 210, 210))])
        assert len(fusion.active_tracks) == 1
        assert fusion.active_tracks[0].frame_count == 2

    def test_process_creates_new_track_for_distant_detection(self):
        fusion = DataFusion()
        fusion.process_detections([_make_detection(bbox=(0, 0, 50, 50))])
        fusion.process_detections([_make_detection(bbox=(400, 400, 500, 500))])
        assert len(fusion.active_tracks) == 2

    def test_get_stable_tracks_respects_min_frames(self):
        fusion = DataFusion(min_track_frames=3)
        for _ in range(2):
            fusion.process_detections([_make_detection(bbox=(100, 100, 200, 200))])
        assert len(fusion.get_stable_tracks()) == 0
        fusion.process_detections([_make_detection(bbox=(105, 105, 205, 205))])
        assert len(fusion.get_stable_tracks()) == 1

    def test_prune_removes_stale_tracks(self):
        fusion = DataFusion(max_track_age=3)
        fusion.process_detections([_make_detection(bbox=(100, 100, 200, 200))])
        for _ in range(5):
            fusion.process_detections([])
        assert len(fusion.active_tracks) == 0

    def test_completed_tracks_cap(self):
        fusion = DataFusion(max_track_age=2, min_track_frames=1)
        fusion.max_completed_tracks = 5
        for i in range(20):
            x = i * 30
            fusion.process_detections([_make_detection(bbox=(x, 0, x + 20, 20))])
            for _ in range(3):
                fusion.process_detections([])
        assert len(fusion.completed_tracks) <= 5

    def test_create_mcp_messages_empty_when_no_stable(self):
        fusion = DataFusion(min_track_frames=3)
        fusion.process_detections([_make_detection()])
        msgs = fusion.create_mcp_messages()
        assert msgs == []

    def test_create_mcp_messages_structure(self):
        fusion = DataFusion(min_track_frames=2)
        det = _make_detection(caption="Wearing red shirt")
        for _ in range(3):
            fusion.process_detections([det])
        msgs = fusion.create_mcp_messages(uav_id="UAV_1")
        assert len(msgs) >= 1
        det_msg = msgs[0]
        assert det_msg["schema"] == "mcp.v0.1"
        assert det_msg["type"] == "MCP.Detection"
        assert det_msg["source"]["system"] == "VITALS"
        assert "payload" in det_msg
        assert det_msg["payload"]["label"] == "person"

    def test_create_mcp_messages_includes_caption(self):
        fusion = DataFusion(min_track_frames=2)
        det = _make_detection(caption="Person waving near debris")
        for _ in range(3):
            fusion.process_detections([det])
        msgs = fusion.create_mcp_messages(include_captions=True)
        types = [m["type"] for m in msgs]
        assert "MCP.Detection" in types
        assert "MCP.Caption" in types

    def test_create_mcp_messages_no_caption_when_none(self):
        fusion = DataFusion(min_track_frames=2)
        det = _make_detection()
        for _ in range(3):
            fusion.process_detections([det])
        msgs = fusion.create_mcp_messages(include_captions=True)
        types = [m["type"] for m in msgs]
        assert "MCP.Caption" not in types

    def test_person_priority_is_high(self):
        fusion = DataFusion(min_track_frames=2)
        det = _make_detection(class_name="person", confidence=0.9)
        for _ in range(3):
            fusion.process_detections([det])
        msgs = fusion.create_mcp_messages()
        det_msg = msgs[0]
        assert det_msg.get("priority", 50) >= 80

    def test_create_mcp_with_geo(self):
        fusion = DataFusion(min_track_frames=2)
        det = _make_detection(location="28.6024,-81.2001")
        for _ in range(3):
            fusion.process_detections([det])
        msgs = fusion.create_mcp_messages()
        det_msg = msgs[0]
        assert "geo" in det_msg
        assert abs(det_msg["geo"]["lat"] - 28.6024) < 0.01
