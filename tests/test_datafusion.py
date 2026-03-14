"""
test_datafusion.py — DataFusion unit tests

Tests temporal tracking, EMA confidence, IoU association, track pruning,
MCP message structure, and priority calculation.
No external services required.

Run:
    python tests/test_datafusion.py
    pytest tests/test_datafusion.py -v
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PerceptionProcessing.Detection import Detection
from DataFusion.Datafusion import DataFusion, TrackedDetection

OUTPUT = ROOT / "tests" / "output"


def _det(class_name="person", confidence=0.8,
         bbox=(100, 100, 300, 300), location=None):
    """Helper: build a synthetic Detection."""
    return Detection(
        class_id=0 if class_name == "person" else 1,
        class_name=class_name,
        confidence=confidence,
        bbox_normalized=(0.1, 0.1, 0.3, 0.3),
        bbox_pixels=bbox,
        location=location,
    )


def _ts():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _header(n, title):
    print(f"\n{'='*70}")
    print(f"  TEST {n}: {title}")
    print(f"{'='*70}")

def _ok(msg):  print(f"  PASS  {msg}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_initialization():
    """1 — DataFusion initialises with correct defaults."""
    _header(1, "Initialization")
    fusion = DataFusion(iou_threshold=0.3, min_track_frames=3, max_track_age=30)
    assert fusion.iou_threshold == 0.3
    assert fusion.min_track_frames == 3
    assert fusion.max_track_age == 30
    assert len(fusion.active_tracks) == 0
    assert fusion.frame_number == 0
    _ok("All default values correct")
    return fusion


def test_new_detection_creates_track():
    """2 — Each new detection creates a unique track."""
    _header(2, "New detection → new track")
    fusion = DataFusion()
    dets = [
        _det("person",  0.8, (100, 100, 300, 300)),
        _det("vehicle", 0.7, (400, 400, 600, 600)),
    ]
    tracks = fusion.process_detections(dets)
    assert len(tracks) == 2
    assert fusion.frame_number == 1
    for t in tracks:
        assert t.track_id.startswith("track_")
        assert t.frame_count == 1
    _ok(f"{len(tracks)} tracks created for {len(dets)} detections")


def test_same_object_updates_track():
    """3 — Same object across frames updates one track, not two."""
    _header(3, "Track continuity across frames")
    fusion = DataFusion(iou_threshold=0.3)
    det = _det("person", 0.8, (100, 100, 300, 300))
    fusion.process_detections([det])
    assert len(fusion.active_tracks) == 1
    initial_id = fusion.active_tracks[0].track_id

    # Same location, same class — should update existing track
    det2 = _det("person", 0.85, (105, 105, 305, 305))
    fusion.process_detections([det2])
    assert len(fusion.active_tracks) == 1, "Should still have 1 track, not 2"
    assert fusion.active_tracks[0].track_id == initial_id
    assert fusion.active_tracks[0].frame_count == 2
    _ok(f"Track {initial_id} updated to frame_count=2")


def test_ema_confidence():
    """4 — Confidence is smoothed with EMA (α=0.3)."""
    _header(4, "EMA confidence smoothing")
    fusion = DataFusion()
    det1 = _det("person", 0.8)
    fusion.process_detections([det1])
    conf_after_1 = fusion.active_tracks[0].confidence
    assert conf_after_1 == 0.8

    det2 = _det("person", 0.9)
    fusion.process_detections([det2])
    expected = 0.7 * 0.8 + 0.3 * 0.9   # α=0.3
    actual = fusion.active_tracks[0].confidence
    assert abs(actual - expected) < 1e-6, f"EMA mismatch: {actual:.6f} vs {expected:.6f}"
    _ok(f"EMA: frame1={conf_after_1:.3f}  +  frame2=0.90  →  {actual:.4f} (expected {expected:.4f})")


def test_iou_same_bbox():
    """5a — IoU of identical boxes is 1.0."""
    _header(5, "IoU calculation")
    det = _det()
    track = TrackedDetection(det, frame_number=1, timestamp=_ts())
    iou = track.iou((100, 100, 300, 300), (100, 100, 300, 300))
    assert abs(iou - 1.0) < 1e-6
    _ok(f"Identical bbox → IoU = {iou:.3f}")

    iou_partial = track.iou((100, 100, 300, 300), (200, 200, 400, 400))
    assert 0.0 < iou_partial < 1.0
    _ok(f"Partial overlap → IoU = {iou_partial:.3f}")

    iou_none = track.iou((100, 100, 200, 200), (300, 300, 400, 400))
    assert iou_none == 0.0
    _ok(f"No overlap → IoU = {iou_none:.3f}")


def test_stable_tracks():
    """6 — get_stable_tracks() only returns tracks seen ≥ min_frames."""
    _header(6, "Stable track detection")
    fusion = DataFusion(min_track_frames=3)
    det = _det("person", 0.8)
    for i in range(5):
        fusion.process_detections([det])
    stable = fusion.get_stable_tracks()
    assert len(stable) == 1
    assert stable[0].frame_count >= 3
    _ok(f"Track has frame_count={stable[0].frame_count} ≥ 3")

    # A fresh fusion with only 2 frames should have no stable tracks
    fusion2 = DataFusion(min_track_frames=3)
    for _ in range(2):
        fusion2.process_detections([det])
    assert len(fusion2.get_stable_tracks()) == 0
    _ok("2-frame track is correctly NOT stable")


def test_track_pruning():
    """7 — Tracks are pruned after max_track_age frames of inactivity."""
    _header(7, "Track pruning")
    fusion = DataFusion(max_track_age=3, min_track_frames=1)
    fusion.process_detections([_det()])
    assert len(fusion.active_tracks) == 1

    # Send empty frames past the age limit
    for _ in range(5):
        fusion.process_detections([])
    assert len(fusion.active_tracks) == 0
    _ok("Track pruned after max_track_age frames of inactivity")


def test_mcp_message_structure():
    """8 — create_mcp_messages() returns valid MCP v0.1 messages."""
    _header(8, "MCP message structure")
    fusion = DataFusion(min_track_frames=1)
    det = _det("person", 0.85, location="28.6024,-81.2001")
    fusion.process_detections([det])

    msgs = fusion.create_mcp_messages(
        tracks=fusion.active_tracks,
        uav_id="UAV_1",
        sector="A1",
        correlation_id="test_mission",
        include_captions=False,
    )
    assert len(msgs) == 1
    msg = msgs[0]

    # Required MCP v0.1 envelope fields
    for field in ("schema", "event_id", "ts", "type", "source", "payload"):
        assert field in msg, f"Missing required field: {field}"
    assert msg["schema"] == "mcp.v0.1"
    assert msg["type"] == "MCP.Detection"

    payload = msg["payload"]
    assert payload["label"] == "person"
    assert "confidence" in payload

    _ok(f"event_id={msg['event_id']}  label={payload['label']}  conf={payload['confidence']:.2f}")

    # Save sample
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "test_datafusion_mcp_sample.json"
    path.write_text(json.dumps(msg, indent=2))
    _ok(f"Sample MCP message saved → {path}")


def test_person_priority():
    """9 — Person detections receive higher priority than other classes."""
    _header(9, "Priority calculation")
    fusion = DataFusion(min_track_frames=1)
    fusion.process_detections([
        _det("person",  0.95, (100, 100, 300, 300)),
        _det("vehicle", 0.95, (400, 400, 600, 600)),
    ])
    msgs = fusion.create_mcp_messages(tracks=fusion.active_tracks, uav_id="UAV_1")
    person_msg  = next(m for m in msgs if m["payload"]["label"] == "person")
    vehicle_msg = next(m for m in msgs if m["payload"]["label"] == "vehicle")
    p_pri = person_msg.get("priority", 50)
    v_pri = vehicle_msg.get("priority", 50)
    assert p_pri > v_pri,  f"Person priority ({p_pri}) should exceed vehicle ({v_pri})"
    assert p_pri >= 80,    f"Person at 95% conf should have priority ≥ 80, got {p_pri}"
    _ok(f"person={p_pri}  vehicle={v_pri}")


def test_caption_included_in_mcp():
    """10 — MCP.Caption message is generated when detection has a caption."""
    _header(10, "Caption included in MCP output")
    fusion = DataFusion(min_track_frames=1)
    det = _det("person", 0.9)
    det.enrich(caption="Person in orange vest waving near debris.")
    fusion.process_detections([det])

    msgs = fusion.create_mcp_messages(
        tracks=fusion.active_tracks,
        uav_id="UAV_1",
        include_captions=True,
    )
    types = [m["type"] for m in msgs]
    assert "MCP.Detection" in types
    assert "MCP.Caption" in types, "Expected MCP.Caption message alongside MCP.Detection"
    caption_msg = next(m for m in msgs if m["type"] == "MCP.Caption")
    assert "caption" in caption_msg["payload"]
    _ok(f"MCP.Caption payload: \"{caption_msg['payload']['caption'][:60]}\"")


def test_multi_drone_tracking():
    """11 — Detections are tracked per drone_id."""
    _header(11, "Multi-drone tracking")
    fusion = DataFusion()
    det = _det("person", 0.8, (100, 100, 300, 300))
    fusion.process_detections([det], drone_id=1)
    fusion.process_detections([det], drone_id=2)
    assert 1 in fusion.drone_tracks
    assert 2 in fusion.drone_tracks
    _ok(f"drone_tracks keys: {list(fusion.drone_tracks.keys())}")


def test_gps_averaging():
    """12 — GPS locations are averaged across frames."""
    _header(12, "GPS location averaging")
    fusion = DataFusion()
    locs = ["28.6020,-81.2000", "28.6022,-81.2002", "28.6024,-81.2004"]
    for loc in locs:
        fusion.process_detections([_det(location=loc)])
    avg = fusion.active_tracks[0].get_average_location()
    assert avg is not None
    lat, lon = [float(x) for x in avg.split(",")]
    assert 28.601 < lat < 28.603, f"Avg lat out of expected range: {lat}"
    _ok(f"Average GPS: {avg}  (from {len(locs)} observations)")


def test_statistics():
    """13 — get_statistics() returns correct counts."""
    _header(13, "Statistics")
    fusion = DataFusion(min_track_frames=3)
    det = _det()
    for _ in range(5):
        fusion.process_detections([det])
    stats = fusion.get_statistics()
    assert stats["frame_number"] == 5
    assert stats["active_tracks"] >= 1
    assert stats["stable_tracks"] >= 1
    _ok(f"stats={stats}")


def test_reset():
    """14 — reset() clears all tracking state."""
    _header(14, "Reset")
    fusion = DataFusion()
    for _ in range(3):
        fusion.process_detections([_det()])
    assert fusion.frame_number == 3
    fusion.reset()
    assert fusion.frame_number == 0
    assert len(fusion.active_tracks) == 0
    _ok("State fully cleared after reset()")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*22 + "DATA FUSION TEST SUITE" + " "*24 + "#")
    print("#"*70)

    tests = [
        ("initialization",           test_initialization),
        ("new detection → track",    test_new_detection_creates_track),
        ("track continuity",         test_same_object_updates_track),
        ("EMA confidence",           test_ema_confidence),
        ("IoU calculation",          test_iou_same_bbox),
        ("stable tracks",            test_stable_tracks),
        ("track pruning",            test_track_pruning),
        ("MCP message structure",    test_mcp_message_structure),
        ("priority calculation",     test_person_priority),
        ("caption in MCP",           test_caption_included_in_mcp),
        ("multi-drone tracking",     test_multi_drone_tracking),
        ("GPS averaging",            test_gps_averaging),
        ("statistics",               test_statistics),
        ("reset",                    test_reset),
    ]

    passed, failed, errors = 0, 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((name, exc))
            import traceback
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()

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
