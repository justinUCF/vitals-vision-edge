"""
test_moondream_prompts.py — Moondream prompt quality evaluation

Runs a ladder of prompts from minimal to complex against every test image
and prints the raw response for each. Use this to find which prompt style
produces clean, SAR-relevant captions before committing to one in production.

Requires Ollama running with moondream pulled:
    ollama serve
    ollama pull moondream

Run:
    python tests/test_moondream_prompts.py
    python tests/test_moondream_prompts.py --host http://192.168.1.50:11434
"""

import sys
import os
import argparse
import base64
import io
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMAGES_DIR = ROOT / "tests" / "images"
OUTPUT_DIR = ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulated bbox: centre third of image (matches production usage pattern)
# Overridden per-image below if you want specific regions.
DEFAULT_BBOX_FRACTION = (1/3, 1/3, 2/3, 2/3)  # (x1%, y1%, x2%, y2%)

# ---------------------------------------------------------------------------
# Prompt ladder — ordered from simplest to most complex
# ---------------------------------------------------------------------------
def build_prompts(x1, y1, x2, y2, img_w, img_h, cx_frac, cy_frac):
    """
    Returns list of (label, prompt) tuples.
    All prompts reference the same bbox so results are comparable.
    """
    h_pos = "left" if cx_frac < 1/3 else ("right" if cx_frac > 2/3 else "center")
    v_pos = "top"  if cy_frac < 1/3 else ("bottom" if cy_frac > 2/3 else "middle")
    rel_pos = f"{v_pos}-{h_pos}"

    return [
        # ── TIER 1: Minimal ──────────────────────────────────────────────────
        (
            "T1-bare",
            "Describe this image in one sentence."
        ),
        (
            "T1-person",
            "Describe the person in this image."
        ),

        # ── TIER 2: Bbox only ────────────────────────────────────────────────
        (
            "T2-bbox-coords",
            f"Describe the person at pixel ({x1},{y1}) to ({x2},{y2})."
        ),
        (
            "T2-bbox-ignore-outside",
            f"Look only at the region from pixel ({x1},{y1}) to ({x2},{y2}). "
            f"Describe the person there in one sentence."
        ),

        # ── TIER 3: Aerial context added ─────────────────────────────────────
        (
            "T3-aerial-bbox",
            f"This is an aerial drone image. "
            f"Describe the person visible between pixel ({x1},{y1}) and ({x2},{y2})."
        ),
        (
            "T3-aerial-rel-pos",
            f"This is an aerial drone image. "
            f"There is a person in the {rel_pos} area of the image "
            f"(pixel region {x1},{y1} to {x2},{y2}). "
            f"Describe their clothing and position in one sentence."
        ),

        # ── TIER 4: Full production-style prompt ─────────────────────────────
        (
            "T4-production-current",
            f"This is an aerial drone image ({img_w}x{img_h} pixels). "
            f"A person is visible in the {rel_pos} of the image, "
            f"within the bounding box from pixel ({x1}, {y1}) to ({x2}, {y2}). "
            f"Ignore everything outside that bounding box. "
            f"In 1-2 sentences, describe only what you can see in that bounding box: "
            f"the person's clothing color, body position, and any visible distress signals. "
            f"Respond in English only."
        ),

        # ── TIER 5: Variations on production prompt ──────────────────────────
        (
            "T5-question-form",
            f"This is a drone aerial image. "
            f"What is the person in the {rel_pos} of the image doing? "
            f"What are they wearing? Answer in 1-2 sentences."
        ),
        (
            "T5-sar-explicit",
            f"Search and rescue aerial image. "
            f"Person detected at {rel_pos} (bbox {x1},{y1} to {x2},{y2}). "
            f"Describe clothing color and any distress signals. One sentence. English only."
        ),
        (
            "T5-no-coords",
            f"This is an aerial drone SAR image. "
            f"There is a person in the {rel_pos} portion of the frame. "
            f"Describe their clothing color, posture, and any visible distress. "
            f"Two sentences maximum. English only."
        ),
    ]


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------
def query_ollama(host: str, model: str, image: Image.Image, prompt: str, timeout: int = 20) -> tuple[str, float]:
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 80},
    }

    t0 = time.perf_counter()
    try:
        r = requests.post(f"{host}/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        elapsed = time.perf_counter() - t0
        return r.json().get("response", "").strip(), elapsed
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama not reachable", 0.0
    except requests.exceptions.Timeout:
        return "ERROR: timeout", float(timeout)
    except Exception as e:
        return f"ERROR: {e}", 0.0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(host: str, model: str):
    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    # Quick connectivity check
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code != 200:
            print(f"Ollama not responding at {host}")
            sys.exit(1)
    except Exception:
        print(f"Cannot reach Ollama at {host}  — start with: ollama serve")
        sys.exit(1)

    results = []  # for JSON dump

    for img_path in images:
        pil = Image.open(img_path).convert("RGB")
        img_w, img_h = pil.size

        # Bbox = centre third of this image
        x1 = int(img_w * DEFAULT_BBOX_FRACTION[0])
        y1 = int(img_h * DEFAULT_BBOX_FRACTION[1])
        x2 = int(img_w * DEFAULT_BBOX_FRACTION[2])
        y2 = int(img_h * DEFAULT_BBOX_FRACTION[3])
        cx_frac = ((x1 + x2) / 2) / img_w
        cy_frac = ((y1 + y2) / 2) / img_h

        prompts = build_prompts(x1, y1, x2, y2, img_w, img_h, cx_frac, cy_frac)

        print(f"\n{'#'*72}")
        print(f"  IMAGE: {img_path.name}  ({img_w}x{img_h})  bbox=({x1},{y1})-({x2},{y2})")
        print(f"{'#'*72}")

        img_results = {"image": img_path.name, "size": [img_w, img_h], "bbox": [x1, y1, x2, y2], "prompts": []}

        for label, prompt in prompts:
            response, elapsed = query_ollama(host, model, pil, prompt)

            print(f"\n  [{label}]  ({elapsed:.1f}s)")
            print(f"  PROMPT : {prompt[:120]}{'...' if len(prompt) > 120 else ''}")
            print(f"  RESPONSE: {response}")

            img_results["prompts"].append({
                "label": label,
                "prompt": prompt,
                "response": response,
                "elapsed_s": round(elapsed, 2),
            })

        results.append(img_results)

    # Save full results to JSON for later review
    out_path = OUTPUT_DIR / "moondream_prompt_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n\nFull results saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moondream prompt ladder evaluation")
    parser.add_argument("--host",  default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--model", default="moondream")
    args = parser.parse_args()

    print(f"Ollama host : {args.host}")
    print(f"Model       : {args.model}")
    print(f"Images      : {IMAGES_DIR}")
    run(args.host, args.model)
