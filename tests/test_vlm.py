"""
Tests for VLMCaptioner — prompt building, image preparation, caption cleaning.

Tests that hit Ollama are skipped when the server is unreachable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PerceptionProcessing.VLMCaptioner import VLMCaptioner

IMAGE = ROOT / "tests" / "images" / "guy_waving.jpg"


@pytest.fixture(scope="module")
def captioner():
    return VLMCaptioner(
        model_name="moondream",
        device="cpu",
        ollama_host="http://localhost:11434",
        timeout=30,
    )


def _ollama_available(host="http://localhost:11434") -> bool:
    try:
        import requests
        r = requests.head(host, timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Initialization ───────────────────────────────────────────────

def test_init_stores_config(captioner):
    assert captioner.model_name == "moondream"
    assert captioner.timeout == 30
    assert captioner.max_image_width == 378


# ── _build_sar_prompt() ─────────────────────────────────────────

def test_prompt_contains_class_name(captioner):
    prompt = captioner._build_sar_prompt("person", (100, 100, 200, 200), 640, 480)
    assert "person" in prompt


def test_prompt_position_center(captioner):
    prompt = captioner._build_sar_prompt("person", (250, 150, 390, 330), 640, 480)
    assert "center" in prompt.lower()


def test_prompt_position_top_left(captioner):
    prompt = captioner._build_sar_prompt("person", (10, 10, 100, 100), 640, 480)
    assert "top" in prompt.lower()
    assert "left" in prompt.lower()


def test_prompt_position_bottom_right(captioner):
    prompt = captioner._build_sar_prompt("person", (500, 400, 630, 470), 640, 480)
    assert "bottom" in prompt.lower()
    assert "right" in prompt.lower()


def test_prompt_no_bbox_defaults_center(captioner):
    prompt = captioner._build_sar_prompt("person")
    assert "center" in prompt.lower()


# ── _prepare_image() ────────────────────────────────────────────

def test_prepare_from_numpy(captioner):
    bgr = np.zeros((100, 200, 3), dtype=np.uint8)
    pil = captioner._prepare_image(bgr)
    assert isinstance(pil, Image.Image)
    assert pil.size == (200, 100)


def test_prepare_from_pil(captioner):
    img = Image.new("RGB", (200, 100))
    result = captioner._prepare_image(img)
    assert isinstance(result, Image.Image)


def test_prepare_from_path(captioner):
    pil = captioner._prepare_image(str(IMAGE))
    assert isinstance(pil, Image.Image)


def test_prepare_invalid_type(captioner):
    with pytest.raises(TypeError):
        captioner._prepare_image(12345)


# ── _clean_caption() ────────────────────────────────────────────

def test_clean_normal_caption(captioner):
    result = captioner._clean_caption("A person in a red shirt standing near debris")
    assert result.endswith(".")
    assert result[0].isupper()


def test_clean_rejects_non_english(captioner):
    result = captioner._clean_caption("これはテスト画像です")
    assert "Error" in result


def test_clean_rejects_garbage(captioner):
    result = captioner._clean_caption("!!!!!!!")
    assert "Error" in result


def test_clean_strips_unwanted_phrases(captioner):
    result = captioner._clean_caption("The image appears to be a person in red")
    assert "image appears" not in result.lower()


def test_clean_truncates_to_two_sentences(captioner):
    result = captioner._clean_caption(
        "First sentence. Second sentence. Third sentence. Fourth sentence."
    )
    assert result.count(". ") <= 1


def test_clean_rejects_heavy_unwanted(captioner):
    text = (
        "The image appears to be captured with a smartphone. "
        "The camera flash is visible in the photo. "
        "The image is low resolution."
    )
    result = captioner._clean_caption(text)
    assert result == "Scene detected, details unclear."


# ── caption_detection() with Ollama ─────────────────────────────

@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_caption_detection_returns_string(captioner):
    import cv2
    frame = cv2.imread(str(IMAGE))
    result = captioner.caption_detection(
        image=frame,
        detection_bbox=(100, 100, 300, 400),
        class_name="person",
        confidence=0.85,
    )
    assert isinstance(result, str)
    assert len(result) > 0


def test_caption_detection_ollama_unavailable():
    cap = VLMCaptioner(ollama_host="http://localhost:99999", timeout=3)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    result = cap.caption_detection(
        image=frame,
        detection_bbox=(10, 10, 50, 50),
        class_name="person",
        confidence=0.8,
    )
    assert "Error" in result


# ── warmup() ────────────────────────────────────────────────────

@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_warmup_succeeds(captioner):
    assert captioner.warmup() is True


def test_warmup_fails_when_unavailable():
    cap = VLMCaptioner(ollama_host="http://localhost:99999", timeout=3)
    assert cap.warmup() is False
