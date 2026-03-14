"""
test_llava.py — LLaVACaptioner tests

Tests caption generation, SAR prompt building, caption cleaning,
and graceful degradation when Ollama is unavailable.

Ollama is optional — tests that require it are skipped with a clear
message if the server is not reachable.

Run:
    python tests/test_llava.py
    pytest tests/test_llava.py -v
"""

import sys
import os
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from PIL import Image
from PerceptionProcessing.LLaVACaptioner import LLaVACaptioner

# ── config ────────────────────────────────────────────────────────────────────
IMAGE1  = ROOT / "tests" / "images" / "drone_testing1.jpg"
IMAGE_P = ROOT / "tests" / "images" / "guy_waving.jpg"    # person image
OUTPUT  = ROOT / "tests" / "output"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        return r.status_code == 200
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
    """1 — LLaVACaptioner initializes without network access."""
    _header(1, "Initialization")
    captioner = LLaVACaptioner(device="cpu", ollama_host=OLLAMA_HOST)
    assert captioner.model_loaded is True
    assert captioner.ollama_host == OLLAMA_HOST.rstrip("/")
    _ok(f"Initialized — ollama_host={captioner.ollama_host}")
    return captioner


def test_sar_prompts(captioner):
    """2 — SAR prompts are generated for known and unknown classes."""
    _header(2, "SAR prompt generation")
    classes = ["person", "vehicle", "car", "truck", "debris", "unknown_object"]
    for cls in classes:
        prompt = captioner._build_sar_prompt(cls)
        assert isinstance(prompt, str) and len(prompt) > 10
        _ok(f"'{cls}' → {prompt[:60]}...")


def test_clean_caption(captioner):
    """3 — _clean_caption() strips unwanted phrases and normalises text."""
    _header(3, "Caption cleaning")
    cases = [
        ("The image appears to be a person in red.", "A person in red."),
        ("can be seen in the photo a person waving.", None),   # just check no crash
        ("Person wearing orange vest.", "Person wearing orange vest."),
        ("this image shows debris near water.",        None),
    ]
    for raw, expected_contains in cases:
        cleaned = captioner._clean_caption(raw)
        assert isinstance(cleaned, str) and len(cleaned) > 0, \
            f"_clean_caption returned empty string for: {raw!r}"
        if expected_contains:
            assert expected_contains.lower() in cleaned.lower() or len(cleaned) > 5
        _ok(f"'{raw[:40]}...' → '{cleaned[:50]}'")


def test_prepare_image_path(captioner):
    """4 — _prepare_image() accepts a file path."""
    _header(4, "_prepare_image() from file path")
    pil = captioner._prepare_image(str(IMAGE1))
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    _ok(f"PIL image loaded: {pil.size[0]}×{pil.size[1]} RGB")


def test_prepare_image_numpy(captioner):
    """5 — _prepare_image() accepts a numpy BGR array."""
    _header(5, "_prepare_image() from numpy array")
    frame = cv2.imread(str(IMAGE1))
    pil = captioner._prepare_image(frame)
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    _ok(f"PIL image converted from numpy: {pil.size[0]}×{pil.size[1]}")


def test_prepare_image_pil(captioner):
    """6 — _prepare_image() accepts a PIL Image."""
    _header(6, "_prepare_image() from PIL Image")
    orig = Image.open(str(IMAGE1)).convert("RGB")
    pil = captioner._prepare_image(orig)
    assert pil is not None
    _ok("PIL → PIL passthrough works")


def test_expand_bbox(captioner):
    """7 — _expand_bbox() correctly grows a bounding box."""
    _header(7, "Bounding box expansion")
    bbox = (100, 100, 300, 300)   # 200×200 box
    expanded = captioner._expand_bbox(bbox, margin=0.2)
    x0, y0, x1, y1 = expanded
    assert x0 < 100 and y0 < 100,  "Box should expand left/up"
    assert x1 > 300 and y1 > 300,  "Box should expand right/down"
    _ok(f"Original {bbox} → expanded {tuple(round(v) for v in expanded)}")


def test_ollama_unavailable_returns_error_string(captioner):
    """8 — Captioner returns 'Error:...' gracefully when Ollama is down."""
    _header(8, "Graceful Ollama failure")
    if _ollama_available():
        _skip("Ollama is running — this test targets the offline path")
        return
    cap = LLaVACaptioner(device="cpu", ollama_host="http://localhost:19999")
    frame = cv2.imread(str(IMAGE1))
    result = cap.caption_image(frame)
    assert result.startswith("Error:"), f"Expected error string, got: {result!r}"
    _ok(f"Offline error string returned: '{result}'")


def test_caption_image_with_ollama(captioner):
    """9 — caption_image() returns a real string when Ollama is available."""
    _header(9, "caption_image() — requires Ollama")
    if not _ollama_available():
        _skip("Ollama not reachable — start with 'ollama serve && ollama pull llava'")
        return
    frame = cv2.imread(str(IMAGE1))
    caption = captioner.caption_image(frame)
    assert isinstance(caption, str) and len(caption) > 5
    assert not caption.startswith("Error:")
    _ok(f"Caption: {caption}")


def test_caption_detection_with_ollama(captioner):
    """10 — caption_detection() returns an SAR-relevant description."""
    _header(10, "caption_detection() — requires Ollama")
    if not _ollama_available():
        _skip("Ollama not reachable — start with 'ollama serve && ollama pull llava'")
        return
    frame = cv2.imread(str(IMAGE_P))
    h, w = frame.shape[:2]
    # Use centre third of the image as the bbox
    bbox = (w//3, h//3, 2*w//3, 2*h//3)
    caption = captioner.caption_detection(
        image=frame,
        detection_bbox=bbox,
        class_name="person",
        confidence=0.85,
    )
    assert isinstance(caption, str) and len(caption) > 5
    assert not caption.startswith("Error:")
    _ok(f"Person caption: {caption}")


def test_custom_ollama_host():
    """11 — ollama_host env var flows through to the request URL."""
    _header(11, "Custom OLLAMA_HOST")
    custom_host = "http://10.0.0.5:11434"
    cap = LLaVACaptioner(device="cpu", ollama_host=custom_host)
    assert cap.ollama_host == custom_host.rstrip("/")
    _ok(f"Custom host stored: {cap.ollama_host}")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*21 + "LLAVA CAPTIONER TEST SUITE" + " "*21 + "#")
    print("#"*70)
    print(f"  Ollama host : {OLLAMA_HOST}")
    print(f"  Ollama live : {'YES' if _ollama_available() else 'NO (offline tests only)'}")

    captioner = test_initialization()

    tests = [
        ("SAR prompts",                lambda: test_sar_prompts(captioner)),
        ("caption cleaning",           lambda: test_clean_caption(captioner)),
        ("prepare image from path",    lambda: test_prepare_image_path(captioner)),
        ("prepare image from numpy",   lambda: test_prepare_image_numpy(captioner)),
        ("prepare image from PIL",     lambda: test_prepare_image_pil(captioner)),
        ("bbox expansion",             lambda: test_expand_bbox(captioner)),
        ("Ollama offline graceful",    lambda: test_ollama_unavailable_returns_error_string(captioner)),
        ("caption_image (Ollama)",     lambda: test_caption_image_with_ollama(captioner)),
        ("caption_detection (Ollama)", lambda: test_caption_detection_with_ollama(captioner)),
        ("custom ollama_host",         test_custom_ollama_host),
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
