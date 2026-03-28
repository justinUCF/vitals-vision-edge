"""
test_moondream_prompts_v2.py — Natural language prompt evaluation (no pixel coords)

Tests prompt variants that use relative position descriptions instead of
pixel coordinates, based on findings from test_moondream_prompts.py which
showed coordinates consistently produce empty or garbage responses.

Requires Ollama running with moondream pulled:
    ollama serve
    ollama pull moondream

Run:
    python tests/test_moondream_prompts_v2.py
    python tests/test_moondream_prompts_v2.py --host http://192.168.1.50:11434
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

IMAGES_DIR = ROOT / "tests" / "images"
OUTPUT_DIR = ROOT / "tests" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BBOX_FRACTION = (1/3, 1/3, 2/3, 2/3)


def rel_pos(cx_frac: float, cy_frac: float) -> str:
    h = "left" if cx_frac < 1/3 else ("right" if cx_frac > 2/3 else "center")
    v = "top"  if cy_frac < 1/3 else ("bottom" if cy_frac > 2/3 else "middle")
    return f"{v}-{h}"


def build_prompts(position: str) -> list[tuple[str, str]]:
    """
    Prompt ladder — all natural language, no pixel coordinates.
    position: e.g. 'middle-center', 'top-left'
    """
    return [
        # ── Baseline (winner from v1) ─────────────────────────────────────────
        (
            "V1-winner-baseline",
            "Describe the person in this image."
        ),

        # ── Add aerial context ────────────────────────────────────────────────
        (
            "aerial-simple",
            "This is an aerial drone image. Describe the person you can see."
        ),
        (
            "aerial-with-position",
            f"This is an aerial drone image. "
            f"Describe the person in the {position} of the frame."
        ),

        # ── Add SAR task framing ──────────────────────────────────────────────
        (
            "sar-position-clothing",
            f"This is an aerial drone image. "
            f"There is a person in the {position} of the frame. "
            f"Describe their clothing color and body position."
        ),
        (
            "sar-position-distress",
            f"This is an aerial drone image. "
            f"There is a person in the {position} of the frame. "
            f"Describe their clothing color, body position, and any visible distress signals. "
            f"Two sentences maximum."
        ),

        # ── Explicit English constraint ───────────────────────────────────────
        (
            "sar-english-only",
            f"This is an aerial drone image. "
            f"There is a person in the {position} of the frame. "
            f"Describe their clothing color, body position, and any visible distress signals. "
            f"Two sentences maximum. Respond in English only."
        ),

        # ── Question form (no coords) ─────────────────────────────────────────
        (
            "question-aerial",
            f"This is an aerial drone image. "
            f"What is the person in the {position} wearing, and what are they doing?"
        ),

        # ── Terse / operator-style ────────────────────────────────────────────
        (
            "terse-operator",
            f"Aerial SAR image. Person visible at {position}. "
            f"Clothing color and posture? One sentence."
        ),
    ]


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


def run(host: str, model: str):
    images = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    if not images:
        print(f"No images found in {IMAGES_DIR}")
        sys.exit(1)

    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code != 200:
            print(f"Ollama not responding at {host}")
            sys.exit(1)
    except Exception:
        print(f"Cannot reach Ollama at {host} — start with: ollama serve")
        sys.exit(1)

    results = []

    for img_path in images:
        pil = Image.open(img_path).convert("RGB")
        img_w, img_h = pil.size

        x1 = int(img_w * DEFAULT_BBOX_FRACTION[0])
        y1 = int(img_h * DEFAULT_BBOX_FRACTION[1])
        x2 = int(img_w * DEFAULT_BBOX_FRACTION[2])
        y2 = int(img_h * DEFAULT_BBOX_FRACTION[3])
        cx_frac = ((x1 + x2) / 2) / img_w
        cy_frac = ((y1 + y2) / 2) / img_h
        position = rel_pos(cx_frac, cy_frac)

        prompts = build_prompts(position)

        print(f"\n{'#'*72}")
        print(f"  IMAGE: {img_path.name}  ({img_w}x{img_h})  position={position}")
        print(f"{'#'*72}")

        img_results = {
            "image": img_path.name,
            "size": [img_w, img_h],
            "position": position,
            "prompts": [],
        }

        for label, prompt in prompts:
            response, elapsed = query_ollama(host, model, pil, prompt)

            status = "EMPTY" if not response else ("ERROR" if response.startswith("ERROR") else "OK")
            print(f"\n  [{label}]  {status}  ({elapsed:.1f}s)")
            print(f"  PROMPT  : {prompt}")
            print(f"  RESPONSE: {response}")

            img_results["prompts"].append({
                "label": label,
                "prompt": prompt,
                "response": response,
                "elapsed_s": round(elapsed, 2),
                "status": status,
            })

        results.append(img_results)

    # Summary table
    prompt_labels = [label for label, _ in build_prompts("middle-center")]
    print(f"\n\n{'='*72}")
    print("  SUMMARY — response rate and quality per prompt")
    print(f"{'='*72}")
    print(f"  {'Prompt':<30}  {'OK':>4}  {'EMPTY':>6}  {'ERROR':>6}")
    print(f"  {'-'*30}  {'-'*4}  {'-'*6}  {'-'*6}")

    for label in prompt_labels:
        ok = empty = error = 0
        for img_result in results:
            for p in img_result["prompts"]:
                if p["label"] == label:
                    if p["status"] == "OK":
                        ok += 1
                    elif p["status"] == "EMPTY":
                        empty += 1
                    else:
                        error += 1
        print(f"  {label:<30}  {ok:>4}  {empty:>6}  {error:>6}")

    out_path = OUTPUT_DIR / "moondream_prompt_eval_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moondream natural-language prompt evaluation")
    parser.add_argument("--host",  default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--model", default="moondream")
    args = parser.parse_args()

    print(f"Ollama host : {args.host}")
    print(f"Model       : {args.model}")
    print(f"Images      : {IMAGES_DIR}")
    run(args.host, args.model)
