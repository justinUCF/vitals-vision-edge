"""
VITALS Vision Edge — main entry point for Jetson Orin Nano.

Pipeline per frame:
  USB Camera → YOLO → DataFusion (temporal tracking)
      → [stable person track, conf ≥ PERSON_THRESH]
          → VLM caption (via Ollama)
              → MCP.Detection + MCP.Caption → TCP to Agent B

A MCP message pair is emitted at most once per tracked person.
Tracks that lose stability or drop below the confidence threshold
are silently skipped until they qualify again.

Environment variables (all optional, defaults shown):
  CAMERA_INDEX       = 0
  YOLO_MODEL         = ../yolo_models/rf3v1.pt  (relative to src/)
  YOLO_CONF          = 0.5    YOLO detection confidence threshold
  PERSON_THRESH      = 0.7    Track EMA confidence required to emit MCP
  CAPTION_THRESH     = 0.7    Per-detection confidence required to call VLM
  MIN_TRACK_FRAMES   = 3      Frames a track must be seen before emitting
  AGENT_B_HOST       = 192.168.1.100
  AGENT_B_PORT       = 9000
  UAV_ID             = UAV_1
  SECTOR             = (none)
  OLLAMA_HOST        = http://localhost:11434
  DEVICE             = cuda   ('cpu' for dev without GPU)
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Ensure src/ is on the path so all sub-packages resolve correctly
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from DataFusion.Datafusion import DataFusion
from Network.TCPSender import TCPSender
from PerceptionProcessing.PerceptionEngine import PerceptionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vitals.main")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent

CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX",     "0"))
YOLO_MODEL       = os.getenv("YOLO_MODEL",       str(_SRC.parent / "yolo_models" / "rf3v1.pt"))
YOLO_CONF        = float(os.getenv("YOLO_CONF",        "0.5"))
PERSON_THRESH    = float(os.getenv("PERSON_THRESH",    "0.7"))
CAPTION_THRESH   = float(os.getenv("CAPTION_THRESH",   "0.7"))
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES",   "3"))

AGENT_B_HOST     = os.getenv("AGENT_B_HOST", "192.168.1.100")
AGENT_B_PORT     = int(os.getenv("AGENT_B_PORT", "9000"))

UAV_ID           = os.getenv("UAV_ID",   "UAV_1")
SECTOR           = os.getenv("SECTOR",   None)
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEVICE           = os.getenv("DEVICE",   "cuda")

GPS_LAT          = os.getenv("GPS_LAT",  None)  # e.g. "28.6024"
GPS_LON          = os.getenv("GPS_LON",  None)  # e.g. "-81.2001"
GPS_LOCATION     = f"{GPS_LAT},{GPS_LON}" if GPS_LAT and GPS_LON else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def open_camera(index: int) -> cv2.VideoCapture:
    # Try V4L2 backend first (Linux / Jetson), fall back to auto-detect
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")

    # Keep the internal frame buffer at 1 to minimise latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera %d opened: %dx%d", index, w, h)
    return cap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Starting VITALS Vision Edge")
    logger.info("YOLO model : %s", YOLO_MODEL)
    logger.info("Ollama host: %s", OLLAMA_HOST)
    logger.info("Agent B    : %s:%d", AGENT_B_HOST, AGENT_B_PORT)
    logger.info("GPS        : %s", GPS_LOCATION or "not set (geo will be omitted from MCP)")

    log_dir = _SRC.parent / "tests" / "output" / "main_mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"mcp_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    mcp_log = open(log_path, "w", encoding="utf-8")
    logger.info("MCP log    : %s", log_path)

    # -- PerceptionEngine (YOLO + VLM) -----------------------------------
    engine = PerceptionEngine(
        yolo_model_path=YOLO_MODEL,
        yolo_confidence=YOLO_CONF,
        caption_threshold=CAPTION_THRESH,
        device=DEVICE,
        yolo_device="cpu",
        ollama_host=OLLAMA_HOST,
    )
    captioner = engine.captioner
    captioner.warmup()

    # -- DataFusion (temporal tracking) ------------------------------------
    fusion = DataFusion(
        iou_threshold=0.3,
        min_track_frames=MIN_TRACK_FRAMES,
        max_track_age=30,
    )

    # -- TCP sender --------------------------------------------------------
    sender = TCPSender(host=AGENT_B_HOST, port=AGENT_B_PORT)

    # -- Camera ------------------------------------------------------------
    cap = open_camera(CAMERA_INDEX)

    # Track IDs that have already had an MCP message emitted so we don't
    # send duplicates for the same physical detection event.
    captioned_tracks: set = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Empty frame received — skipping")
                time.sleep(0.05)
                continue

            # 1. YOLO on the raw frame (captions handled manually below)
            detections, _ = engine.process_image(
                frame,
                drone_id=1,
                location=GPS_LOCATION,
                generate_captions=False,
                return_annotated=False,
            )

            # 2. Feed into DataFusion for temporal tracking
            fusion.process_detections(detections, drone_id=1)

            # 3. Check every stable track for qualifying persons
            for track in fusion.get_stable_tracks():
                # Only care about people
                if track.class_name.lower() != "person":
                    continue

                # EMA-smoothed confidence must meet the bar
                if track.confidence < PERSON_THRESH:
                    continue

                # One MCP emission per track
                if track.track_id in captioned_tracks:
                    continue

                # 4. Pick the highest-confidence detection for captioning
                best_det = track.get_best_detection()
                if best_det.confidence < CAPTION_THRESH:
                    continue

                logger.info(
                    "Qualifying person track %s — conf=%.2f frames=%d — requesting caption",
                    track.track_id, track.confidence, track.frame_count,
                )

                # 5. VLM caption
                try:
                    caption = captioner.caption_detection(
                        image=frame,
                        detection_bbox=best_det.bbox_pixels,
                        class_name=best_det.class_name,
                        confidence=best_det.confidence,
                    )
                except Exception as exc:
                    logger.error("VLM caption failed: %s", exc)
                    continue

                if not caption or caption.startswith("Error:"):
                    logger.warning("Caption unavailable (%s) — using fallback", caption or "empty")
                    caption = "No caption could be generated."

                # Attach caption so create_mcp_messages includes MCP.Caption
                best_det.enrich(caption=caption)

                # 6. Build MCP.Detection + MCP.Caption pair
                messages = fusion.create_mcp_messages(
                    tracks=[track],
                    uav_id=UAV_ID,
                    sector=SECTOR,
                    include_captions=True,
                )

                # 7. Send each message over TCP and log to file
                for msg in messages:
                    sender.send(msg)
                    mcp_log.write(json.dumps(msg) + "\n")
                    mcp_log.flush()
                    logger.info(
                        "MCP %s [%s] track=%s conf=%.2f — queued (buffer=%d)",
                        msg.get("type", "?"),
                        msg.get("event_id", "?"),
                        track.track_id,
                        track.confidence,
                        sender.buffer_size(),
                    )

                captioned_tracks.add(track.track_id)

    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        cap.release()
        sender.close()
        mcp_log.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
