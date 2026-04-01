"""
Tests for MCP v0.1 schema — MCPDetection, MCPCaption, and the
create_detection_from_yolo helper.

No external services required.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from protocols.mcp_schema import MCPDetection, MCPCaption, create_detection_from_yolo
from PerceptionProcessing.Detection import Detection


# ── MCPDetection ─────────────────────────────────────────────────

class TestMCPDetection:

    def test_required_fields(self):
        det = MCPDetection(label="person", confidence=0.9)
        d = det.to_dict()
        assert d["schema"] == "mcp.v0.1"
        assert d["type"] == "MCP.Detection"
        assert d["event_id"].startswith("evt_")
        assert "ts" in d
        assert d["source"]["system"] == "VITALS"
        assert d["payload"]["label"] == "person"
        assert d["payload"]["confidence"] == 0.9

    def test_event_id_unique(self):
        a = MCPDetection(label="person", confidence=0.8)
        b = MCPDetection(label="person", confidence=0.8)
        assert a.event_id != b.event_id

    def test_priority_clamped_high(self):
        det = MCPDetection(label="person", confidence=0.9, priority=150)
        assert det.priority == 100

    def test_priority_clamped_low(self):
        det = MCPDetection(label="person", confidence=0.9, priority=-10)
        assert det.priority == 0

    def test_priority_default(self):
        det = MCPDetection(label="person", confidence=0.9)
        d = det.to_dict()
        # Default priority is 50, which is omitted from dict
        assert "priority" not in d

    def test_geo_included_when_coordinates_set(self):
        det = MCPDetection(
            label="person", confidence=0.9,
            latitude=28.6024, longitude=-81.2001,
        )
        d = det.to_dict()
        assert "geo" in d
        assert d["geo"]["lat"] == 28.6024
        assert d["geo"]["lon"] == -81.2001

    def test_geo_omitted_when_no_coordinates(self):
        det = MCPDetection(label="person", confidence=0.9)
        d = det.to_dict()
        assert "geo" not in d

    def test_bbox_included(self):
        det = MCPDetection(
            label="person", confidence=0.9,
            bbox_xyxy=[100.0, 200.0, 300.0, 500.0],
        )
        d = det.to_dict()
        assert d["payload"]["bbox_xyxy"] == [100.0, 200.0, 300.0, 500.0]

    def test_tags_included(self):
        det = MCPDetection(
            label="person", confidence=0.9,
            tags=["stable_track", "person_detected"],
        )
        d = det.to_dict()
        assert "stable_track" in d["tags"]

    def test_text_auto_generated(self):
        det = MCPDetection(label="person", confidence=0.92, uav_id="UAV_1")
        d = det.to_dict()
        assert "text" in d
        assert "Person" in d["text"]
        assert "92%" in d["text"]

    def test_correlation_id(self):
        det = MCPDetection(
            label="person", confidence=0.9, correlation_id="mission_001"
        )
        d = det.to_dict()
        assert d["corr_id"] == "mission_001"


# ── MCPCaption ───────────────────────────────────────────────────

class TestMCPCaption:

    def test_required_fields(self):
        cap = MCPCaption(caption="Person in red shirt waving")
        d = cap.to_dict()
        assert d["schema"] == "mcp.v0.1"
        assert d["type"] == "MCP.Caption"
        assert d["event_id"].startswith("evt_")
        assert d["payload"]["caption"] == "Person in red shirt waving"

    def test_text_is_caption(self):
        cap = MCPCaption(caption="Person in red shirt waving")
        d = cap.to_dict()
        assert d["text"] == "Person in red shirt waving"

    def test_optional_fields(self):
        cap = MCPCaption(
            caption="test",
            confidence=0.85,
            sector="B4",
            uav_id="UAV_2",
            correlation_id="mission_001",
        )
        d = cap.to_dict()
        assert d["payload"]["confidence"] == 0.85
        assert d["payload"]["sector"] == "B4"
        assert d["corr_id"] == "mission_001"

    def test_event_id_unique(self):
        a = MCPCaption(caption="a")
        b = MCPCaption(caption="b")
        assert a.event_id != b.event_id


# ── create_detection_from_yolo() ─────────────────────────────────

class TestCreateDetectionFromYolo:

    def _make_detection(self, **kwargs):
        defaults = dict(
            class_id=0, class_name="person", confidence=0.85,
            bbox_normalized=(0.1, 0.2, 0.5, 0.8),
            bbox_pixels=(64, 96, 320, 384),
        )
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_basic_conversion(self):
        det = self._make_detection()
        mcp = create_detection_from_yolo(det, frame_number=10, uav_id="UAV_1")
        assert isinstance(mcp, MCPDetection)
        assert mcp.label == "person"
        assert mcp.confidence == 0.85
        assert mcp.bbox_xyxy == [64.0, 96.0, 320.0, 384.0]
        assert mcp.uav_id == "UAV_1"

    def test_with_gps(self):
        det = self._make_detection(location="28.6024,-81.2001")
        mcp = create_detection_from_yolo(det, frame_number=1, uav_id="UAV_1")
        assert abs(mcp.latitude - 28.6024) < 0.001
        assert abs(mcp.longitude - (-81.2001)) < 0.001

    def test_without_gps(self):
        det = self._make_detection()
        mcp = create_detection_from_yolo(det, frame_number=1, uav_id="UAV_1")
        assert mcp.latitude is None
        assert mcp.longitude is None

    def test_sector_and_correlation(self):
        det = self._make_detection()
        mcp = create_detection_from_yolo(
            det, frame_number=1, uav_id="UAV_1",
            sector="B4", correlation_id="mission_001",
        )
        assert mcp.sector == "B4"
        assert mcp.correlation_id == "mission_001"
