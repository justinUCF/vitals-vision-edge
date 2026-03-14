"""
test_perceptionengine.py — PerceptionEngine integration tests

Tests the YOLO + LLaVA orchestration layer.
LLaVA tests are skipped gracefully when Ollama is not running.

Run:
    python tests/test_perceptionengine.py
    pytest tests/test_perceptionengine.py -v
"""

import sys
import json
import os
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PerceptionProcessing.PerceptionEngine import PerceptionEngine
from PerceptionProcessing.Detection import Detection

# ── config ────────────────────────────────────────────────────────────────────
MODEL       = ROOT / "yolo_models" / "rf3v1.pt"
IMAGE       = ROOT / "tests" / "images" / "drone_testing1.jpg"
IMAGE_P     = ROOT / "tests" / "images" / "guy_waving.jpg"
OUTPUT      = ROOT / "tests" / "output"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _ollama_available():
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def _header(n, title):
    print(f"\n{'='*70}")
    print(f"  TEST {n}: {title}")
    print(f"{'='*70}")

def _ok(msg):   print(f"  PASS  {msg}")
def _skip(msg): print(f"  SKIP  {msg}")


# ── tests ─────────────────────────────────────────────────────────────────────

def test_initialization():
    """1 — Engine initialises with both components ready."""
    _header(1, "Initialization")
    engine = PerceptionEngine(
        yolo_model_path=str(MODEL),
        yolo_confidence=0.5,
        caption_threshold=0.7,
        device="cpu",
        ollama_host=OLLAMA_HOST,
    )
    assert engine.yolo_detector is not None
    assert engine.captioner is not None
    assert engine.caption_threshold == 0.7
    assert engine.captioner.ollama_host == OLLAMA_HOST.rstrip("/")
    _ok(f"Engine ready — caption_threshold={engine.caption_threshold}  ollama_host={engine.captioner.ollama_host}")
    return engine


def test_detect_from_path_no_captions(engine):
    """2 — process_image_path() without captions."""
    _header(2, "process_image_path() — no captions")
    dets, annotated = engine.process_image_path(
        str(IMAGE),
        generate_captions=False,
        return_annotated=True,
    )
    assert len(dets) > 0,       "Test image should produce at least one detection"
    assert annotated is not None
    for d in dets:
        assert d.caption is None, "Captions should be None when generate_captions=False"
    _ok(f"{len(dets)} detection(s)  annotated={annotated.shape}")


def test_detect_numpy_array(engine):
    """3 — process_image() accepts a raw numpy BGR frame."""
    _header(3, "process_image() — numpy array input")
    frame = cv2.imread(str(IMAGE))
    dets, annotated = engine.process_image(
        frame,
        drone_id=1,
        generate_captions=False,
        return_annotated=True,
    )
    assert len(dets) > 0
    assert annotated.shape == frame.shape
    _ok(f"numpy path: {len(dets)} detection(s)")


def test_drone_metadata_enrichment(engine):
    """4 — drone_id and location are attached to every detection."""
    _header(4, "Drone metadata enrichment")
    dets, _ = engine.process_image_path(
        str(IMAGE),
        drone_id=42,
        location="28.6024,-81.2001",
        generate_captions=False,
        return_annotated=False,
    )
    assert len(dets) > 0
    for d in dets:
        assert d.drone_id == 42,             f"Expected drone_id=42, got {d.drone_id}"
        assert d.location == "28.6024,-81.2001", f"Location mismatch: {d.location}"
    _ok(f"All {len(dets)} detections enriched with drone_id=42, location='28.6024,-81.2001'")


def test_no_annotated_when_false(engine):
    """5 — return_annotated=False returns None."""
    _header(5, "return_annotated=False")
    _, annotated = engine.process_image_path(
        str(IMAGE), generate_captions=False, return_annotated=False,
    )
    assert annotated is None
    _ok("annotated image is None as expected")


def test_confidence_threshold_respected(engine):
    """6 — Detections below caption_threshold get no caption even with generate_captions=True."""
    _header(6, "Caption threshold filtering")
    if not _ollama_available():
        _skip("Ollama not running — caption filtering test requires it")
        return
    strict_engine = PerceptionEngine(
        yolo_model_path=str(MODEL),
        yolo_confidence=0.3,    # detect low-confidence objects
        caption_threshold=0.99, # only caption near-perfect detections
        device="cpu",
        ollama_host=OLLAMA_HOST,
    )
    dets, _ = strict_engine.process_image_path(
        str(IMAGE), generate_captions=True, return_annotated=False,
    )
    for d in dets:
        if d.confidence < 0.99:
            assert d.caption is None, \
                f"Detection conf={d.confidence:.2f} should not have a caption (threshold=0.99)"
    below = sum(1 for d in dets if d.confidence < 0.99)
    _ok(f"{below} detections below threshold=0.99 have no caption")


def test_caption_generated_when_ollama_up(engine):
    """7 — generate_captions=True produces captions when Ollama is running."""
    _header(7, "Caption generation — requires Ollama")
    if not _ollama_available():
        _skip("Ollama not running — start with 'ollama serve && ollama pull llava'")
        return
    # Use the image we know yields detections; lower threshold to be safe
    low_thresh_engine = PerceptionEngine(
        yolo_model_path=str(MODEL),
        yolo_confidence=0.3,
        caption_threshold=0.3,
        device="cpu",
        ollama_host=OLLAMA_HOST,
    )
    dets, _ = low_thresh_engine.process_image_path(
        str(IMAGE),
        generate_captions=True,
        return_annotated=False,
    )
    if not dets:
        _skip("No detections in test image at conf=0.3 — cannot test captioning")
        return
    captioned = [d for d in dets if d.caption is not None and not d.caption.startswith("Error:")]
    assert len(captioned) > 0, "Expected at least one successful caption"
    for d in captioned:
        _ok(f"  {d.class_name} ({d.confidence:.0%}) → {d.caption[:70]}")


def test_to_dict_serialization(engine):
    """8 — Detection.to_dict() produces valid JSON-serialisable output."""
    _header(8, "Detection JSON serialisation")
    dets, _ = engine.process_image_path(
        str(IMAGE), drone_id=7, location="28.5,-81.5",
        generate_captions=False, return_annotated=False,
    )
    records = [d.to_dict() for d in dets]
    json_str = json.dumps(records)          # should not raise
    reloaded = json.loads(json_str)
    assert len(reloaded) == len(dets)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "test_perception_detections.json"
    out.write_text(json.dumps(records, indent=2))
    _ok(f"Serialised {len(records)} detections → {out}")


def test_batch_consistency(engine):
    """9 — Same image fed multiple times produces consistent detection counts."""
    _header(9, "Batch consistency")
    counts = []
    for i in range(5):
        dets, _ = engine.process_image_path(
            str(IMAGE), generate_captions=False, return_annotated=False,
        )
        counts.append(len(dets))
    assert len(set(counts)) == 1, f"Inconsistent counts across frames: {counts}"
    _ok(f"5 identical frames → {counts[0]} detections each time")


def test_process_person_image(engine):
    """10 — guy_waving.jpg runs through the engine and saves an annotated image."""
    _header(10, "Process person image (guy_waving.jpg)")
    # Use a lower threshold so the image is more likely to yield detections
    low_engine = PerceptionEngine(
        yolo_model_path=str(MODEL),
        yolo_confidence=0.3,
        device="cpu",
        ollama_host=OLLAMA_HOST,
    )
    dets, annotated = low_engine.process_image_path(
        str(IMAGE_P), drone_id=1, generate_captions=False, return_annotated=True,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = OUTPUT / "test_perception_annotated.jpg"
    cv2.imwrite(str(out), annotated)
    person_dets = [d for d in dets if d.class_name.lower() == "person"]
    _ok(f"{len(dets)} total detections, {len(person_dets)} person(s) — saved → {out}")
    if len(dets) == 0:
        _skip("Model found no objects in guy_waving.jpg at conf=0.3 (model may not cover this scene)")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*19 + "PERCEPTION ENGINE TEST SUITE" + " "*21 + "#")
    print("#"*70)
    print(f"  Model       : {MODEL}")
    print(f"  Ollama live : {'YES' if _ollama_available() else 'NO (caption tests skipped)'}")

    engine = test_initialization()

    tests = [
        ("detect from path",          lambda: test_detect_from_path_no_captions(engine)),
        ("numpy array input",         lambda: test_detect_numpy_array(engine)),
        ("drone metadata",            lambda: test_drone_metadata_enrichment(engine)),
        ("no annotated image",        lambda: test_no_annotated_when_false(engine)),
        ("caption threshold",         lambda: test_confidence_threshold_respected(engine)),
        ("caption with Ollama",       lambda: test_caption_generated_when_ollama_up(engine)),
        ("JSON serialisation",        lambda: test_to_dict_serialization(engine)),
        ("batch consistency",         lambda: test_batch_consistency(engine)),
        ("person image",              lambda: test_process_person_image(engine)),
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
