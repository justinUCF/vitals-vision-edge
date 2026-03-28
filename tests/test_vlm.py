"""
test_vlm.py — VLMCaptioner tests

Tests caption generation, SAR prompt building, caption cleaning,
and graceful degradation when Ollama is unavailable.

Ollama is optional — tests that require it are skipped with a clear
message if the server is not reachable.

Run:
    python tests/test_vlm.py
    pytest tests/test_vlm.py -v
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
from PerceptionProcessing.VLMCaptioner import VLMCaptioner

# ── config ────────────────────────────────────────────────────────────────────
IMAGES_DIR  = ROOT / "tests" / "images"
IMAGES      = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
OUTPUT      = ROOT / "tests" / "output"
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
    """1 — VLMCaptioner initializes without network access."""
    _header(1, "Initialization")
    captioner = VLMCaptioner(device="cpu", ollama_host=OLLAMA_HOST)
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
        assert "pixel" not in prompt.lower(), \
            "Prompt should not contain pixel coordinates (causes empty responses)"
        _ok(f"'{cls}' → {prompt[:80]}...")


def test_sar_prompt_uses_relative_position(captioner):
    """3 — _build_sar_prompt() uses relative position when bbox and dims given."""
    _header(3, "SAR prompt relative position")
    # Top-left detection (cx=50, cy=50 in a 640x640 image)
    prompt = captioner._build_sar_prompt("person", bbox=(10, 10, 90, 90), img_w=640, img_h=640)
    assert "top-left" in prompt, f"Expected 'top-left' in prompt, got: {prompt}"
    assert "pixel" not in prompt.lower()
    _ok(f"top-left: {prompt}")

    # Bottom-right
    prompt = captioner._build_sar_prompt("person", bbox=(500, 500, 620, 620), img_w=640, img_h=640)
    assert "bottom-right" in prompt, f"Expected 'bottom-right' in prompt, got: {prompt}"
    _ok(f"bottom-right: {prompt}")

    # Middle-center
    prompt = captioner._build_sar_prompt("person", bbox=(200, 200, 440, 440), img_w=640, img_h=640)
    assert "middle-center" in prompt, f"Expected 'middle-center' in prompt, got: {prompt}"
    _ok(f"middle-center: {prompt}")


def test_clean_caption(captioner):
    """4 — _clean_caption() strips unwanted phrases and normalises text."""
    _header(4, "Caption cleaning")
    cases = [
        ("The image appears to be a person in red.", None),
        ("can be seen in the photo a person waving.", None),
        ("Person wearing orange vest.", "Person wearing orange vest."),
        ("this image shows debris near water.", None),
    ]
    for raw, expected_exact in cases:
        cleaned = captioner._clean_caption(raw)
        assert isinstance(cleaned, str) and len(cleaned) > 0, \
            f"_clean_caption returned empty string for: {raw!r}"
        if expected_exact:
            assert cleaned == expected_exact, f"Expected '{expected_exact}', got '{cleaned}'"
        _ok(f"'{raw[:40]}' → '{cleaned[:50]}'")


def test_clean_caption_rejects_garbage(captioner):
    """5 — _clean_caption() rejects non-English and all-punctuation output."""
    _header(5, "Caption cleaning — garbage rejection")
    # Thai text like the hallucination seen in production
    thai = "\u0e23\u0e16\u0e32\u0e40\u0e21\u0e35\u0e22\u0e07\u0e15\u0e31\u0e2a\u0e14\u0e35"
    result = captioner._clean_caption(thai)
    assert result.startswith("Error:"), f"Expected Error: for non-ASCII, got: {result!r}"
    _ok(f"Non-ASCII correctly rejected: '{result}'")

    # All-punctuation garbage like "!!!!" or "!!!YOU ARE HERE!!!"
    for garbage in ["!!!!", "!!!!!!!!!!!!!!!!", "!?!?!?!?!?"]:
        result = captioner._clean_caption(garbage)
        assert result.startswith("Error:"), \
            f"Expected Error: for all-punctuation '{garbage}', got: {result!r}"
    _ok("All-punctuation garbage correctly rejected")


def test_prepare_image_path(captioner):
    """6 — _prepare_image() accepts a file path."""
    _header(6, "_prepare_image() from file path")
    pil = captioner._prepare_image(str(IMAGES[0]))
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    _ok(f"PIL image loaded: {pil.size[0]}×{pil.size[1]} RGB")


def test_prepare_image_numpy(captioner):
    """7 — _prepare_image() accepts a numpy BGR array."""
    _header(7, "_prepare_image() from numpy array")
    frame = cv2.imread(str(IMAGES[0]))
    pil = captioner._prepare_image(frame)
    assert isinstance(pil, Image.Image)
    assert pil.mode == "RGB"
    _ok(f"PIL image converted from numpy: {pil.size[0]}×{pil.size[1]}")


def test_prepare_image_pil(captioner):
    """8 — _prepare_image() accepts a PIL Image."""
    _header(8, "_prepare_image() from PIL Image")
    orig = Image.open(str(IMAGES[0])).convert("RGB")
    pil = captioner._prepare_image(orig)
    assert pil is not None
    _ok("PIL → PIL passthrough works")


def test_ollama_unavailable_returns_error_string(captioner):
    """9 — Captioner returns 'Error:...' gracefully when Ollama is down."""
    _header(9, "Graceful Ollama failure")
    if _ollama_available():
        _skip("Ollama is running — this test targets the offline path")
        return
    cap = VLMCaptioner(device="cpu", ollama_host="http://localhost:19999")
    frame = cv2.imread(str(IMAGES[0]))
    result = cap.caption_image(frame)
    assert result.startswith("Error:"), f"Expected error string, got: {result!r}"
    _ok(f"Offline error string returned: '{result}'")


def test_caption_detection_all_images(captioner):
    """10 — caption_detection() runs on every test image when Ollama is available."""
    _header(10, f"caption_detection() — all {len(IMAGES)} images — requires Ollama")
    if not _ollama_available():
        _skip("Ollama not reachable — start with 'ollama serve && ollama pull moondream'")
        return

    results = {"ok": 0, "empty": 0, "error": 0}

    for img_path in IMAGES:
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        bbox = (w // 3, h // 3, 2 * w // 3, 2 * h // 3)

        caption = captioner.caption_detection(
            image=frame,
            detection_bbox=bbox,
            class_name="person",
            confidence=0.8,
        )

        assert isinstance(caption, str), \
            f"{img_path.name}: caption_detection must return a string, got {type(caption)}"

        if not caption or caption.startswith("Error:"):
            results["empty" if not caption else "error"] += 1
            print(f"  WARN  {img_path.name}: '{caption or '(empty)'}'")
        else:
            results["ok"] += 1
            _ok(f"{img_path.name}: '{caption[:70]}'")

    print(f"\n  Summary: {results['ok']} captioned, "
          f"{results['empty']} empty, {results['error']} error "
          f"across {len(IMAGES)} images")

    # sar-position-clothing prompt is expected to get 5/6 or better
    min_ok = max(1, len(IMAGES) - 1)
    assert results["ok"] >= min_ok, \
        f"Expected captions on at least {min_ok}/{len(IMAGES)} images, got {results['ok']}"


def test_custom_ollama_host():
    """11 — Custom ollama_host flows through to the stored attribute."""
    _header(11, "Custom OLLAMA_HOST")
    custom_host = "http://10.0.0.5:11434"
    cap = VLMCaptioner(device="cpu", ollama_host=custom_host)
    assert cap.ollama_host == custom_host.rstrip("/")
    _ok(f"Custom host stored: {cap.ollama_host}")


# ── runner ────────────────────────────────────────────────────────────────────

def run():
    print("\n" + "#"*70)
    print("#" + " "*22 + "VLM CAPTIONER TEST SUITE" + " "*22 + "#")
    print("#"*70)
    print(f"  Ollama host : {OLLAMA_HOST}")
    print(f"  Ollama live : {'YES' if _ollama_available() else 'NO (offline tests only)'}")
    print(f"  Test images : {len(IMAGES)} files in tests/images/")

    captioner = test_initialization()

    tests = [
        ("SAR prompt generation",       lambda: test_sar_prompts(captioner)),
        ("SAR prompt relative pos",     lambda: test_sar_prompt_uses_relative_position(captioner)),
        ("caption cleaning",            lambda: test_clean_caption(captioner)),
        ("garbage rejection",            lambda: test_clean_caption_rejects_garbage(captioner)),
        ("prepare image from path",     lambda: test_prepare_image_path(captioner)),
        ("prepare image from numpy",    lambda: test_prepare_image_numpy(captioner)),
        ("prepare image from PIL",      lambda: test_prepare_image_pil(captioner)),
        ("Ollama offline graceful",     lambda: test_ollama_unavailable_returns_error_string(captioner)),
        ("caption_detection all images",lambda: test_caption_detection_all_images(captioner)),
        ("custom ollama_host",          test_custom_ollama_host),
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
