"""
demo_live_pipeline.py — Visual demonstration of the full VITALS pipeline.

Identical to src/main.py but renders an OpenCV window showing the live
annotated camera feed (bounding boxes + labels) so you can visually confirm
the pipeline is working end-to-end:

  USB Camera → YOLO → DataFusion → VLM caption → MCP.Detection + MCP.Caption → TCP to Agent B

Press Q to quit.

All configuration is via the same environment variables as src/main.py:
  CAMERA_INDEX, YOLO_MODEL, YOLO_CONF, PERSON_THRESH, CAPTION_THRESH,
  MIN_TRACK_FRAMES, AGENT_B_HOST, AGENT_B_PORT, UAV_ID, SECTOR,
  OLLAMA_HOST, DEVICE, GPS_LAT, GPS_LON
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from DataFusion.Datafusion import DataFusion
from Network.TCPSender import TCPSender
from PerceptionProcessing.PerceptionEngine import PerceptionEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vitals.demo")

# ---------------------------------------------------------------------------
# Configuration — identical to src/main.py
# ---------------------------------------------------------------------------
_SRC = ROOT / "src"

CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX",     "0"))
YOLO_MODEL       = os.getenv("YOLO_MODEL",       str(ROOT / "yolo_models" / "rf3v1.pt"))
YOLO_CONF        = float(os.getenv("YOLO_CONF",        "0.5"))
PERSON_THRESH    = float(os.getenv("PERSON_THRESH",    "0.7"))
CAPTION_THRESH   = float(os.getenv("CAPTION_THRESH",   "0.7"))
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES",   "3"))

AGENT_B_HOST     = os.getenv("AGENT_B_HOST", "192.168.1.100")
AGENT_B_PORT     = int(os.getenv("AGENT_B_PORT", "9000"))

UAV_ID           = os.getenv("UAV_ID",   "UAV_1")
SECTOR           = os.getenv("SECTOR",   None)
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEVICE           = os.getenv("DEVICE",   "cpu")

GPS_LAT          = os.getenv("GPS_LAT",  25)
GPS_LON          = os.getenv("GPS_LON",  25)
GPS_LOCATION     = f"{GPS_LAT},{GPS_LON}" if GPS_LAT and GPS_LON else None


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera at index {index}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera %d opened: %dx%d", index, w, h)
    return cap


def open_mcp_log() -> tuple:
    """Open a timestamped JSONL file for MCP message logging. Returns (path, file handle)."""
    log_dir = ROOT / "tests" / "output" / "demo_mcp"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"mcp_log_{ts}.jsonl"
    return log_path, open(log_path, "w", encoding="utf-8")


def main():
    logger.info("Starting VITALS Vision Edge (demo)")
    logger.info("YOLO model : %s", YOLO_MODEL)
    logger.info("Ollama host: %s", OLLAMA_HOST)
    logger.info("Agent B    : %s:%d", AGENT_B_HOST, AGENT_B_PORT)
    logger.info("GPS        : %s", GPS_LOCATION or "not set (geo will be omitted from MCP)")

    log_path, mcp_log = open_mcp_log()
    logger.info("MCP log    : %s", log_path)

    engine = PerceptionEngine(
        yolo_model_path=YOLO_MODEL,
        yolo_confidence=YOLO_CONF,
        caption_threshold=CAPTION_THRESH,
        device=DEVICE,
        ollama_host=OLLAMA_HOST,
    )
    captioner = engine.captioner

    fusion = DataFusion(
        iou_threshold=0.3,
        min_track_frames=MIN_TRACK_FRAMES,
        max_track_age=30,
    )

    sender = TCPSender(host=AGENT_B_HOST, port=AGENT_B_PORT)
    cap = open_camera(CAMERA_INDEX)
    captioned_tracks: set = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Empty frame received — skipping")
                time.sleep(0.05)
                continue

            # 1. YOLO — request annotated frame for display
            detections, annotated = engine.process_image(
                frame,
                drone_id=1,
                location=GPS_LOCATION,
                generate_captions=False,
                return_annotated=True,
            )

            # 2. DataFusion temporal tracking
            fusion.process_detections(detections, drone_id=1)

            # 3. Check stable tracks for qualifying persons
            for track in fusion.get_stable_tracks():
                if track.class_name.lower() != "person":
                    continue
                if track.confidence < PERSON_THRESH:
                    continue
                if track.track_id in captioned_tracks:
                    continue

                best_det = track.get_best_detection()
                if best_det.confidence < CAPTION_THRESH:
                    continue

                logger.info(
                    "Qualifying person track %s — conf=%.2f frames=%d — requesting caption",
                    track.track_id, track.confidence, track.frame_count,
                )

                # 4. VLM caption
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
                    logger.warning("Caption invalid — skipping MCP: %s", caption)
                    continue

                best_det.enrich(caption=caption)

                # 5. Build and send MCP messages
                messages = fusion.create_mcp_messages(
                    tracks=[track],
                    uav_id=UAV_ID,
                    sector=SECTOR,
                    include_captions=True,
                )

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

                logger.info(
                    "Caption confirmed for track %s: %s",
                    track.track_id, caption[:100] + ("..." if len(caption) > 100 else ""),
                )
                captioned_tracks.add(track.track_id)

            # 6. Show annotated frame
            display = annotated if annotated is not None else frame
            cv2.imshow("VITALS Vision Edge", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Q pressed — shutting down")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted — shutting down")
    finally:
        cap.release()
        sender.close()
        mcp_log.close()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
