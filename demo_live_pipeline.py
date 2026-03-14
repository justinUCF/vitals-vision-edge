"""
demo_live_pipeline.py — Live USB camera demonstration
=====================================================

Shows the full VITALS Vision Edge pipeline running in real time:

  USB Camera → YOLO detection → DataFusion tracking
      → [stable person, conf ≥ 0.7] → LLaVA caption (async)
          → MCP.Detection + MCP.Caption → TCP to Agent B

Display layout
──────────────
┌─────────────────────────┬──────────────────────┐
│                         │  PIPELINE STATUS      │
│   Live YOLO feed        │  Tracks / Thresholds  │
│   (bounding boxes)      │  LLaVA status         │
│                         │  Last MCP message     │
└─────────────────────────┴──────────────────────┘

Controls
────────
  Q   — quit
  R   — reset DataFusion tracking state
  S   — save current frame + status panel to tests/output/

Usage
─────
  python demo_live_pipeline.py [--camera 0] [--agent-b 192.168.1.100:9000]

All options can also be set via environment variables (same as src/main.py).
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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

# ── defaults (all overridable via env or CLI) ─────────────────────────────────
CAMERA_INDEX     = int(os.getenv("CAMERA_INDEX",     "0"))
YOLO_MODEL       = os.getenv("YOLO_MODEL",       str(ROOT / "yolo_models" / "rf3v1.pt"))
YOLO_CONF        = float(os.getenv("YOLO_CONF",        "0.5"))
PERSON_THRESH    = float(os.getenv("PERSON_THRESH",    "0.7"))
CAPTION_THRESH   = float(os.getenv("CAPTION_THRESH",   "0.7"))
MIN_TRACK_FRAMES = int(os.getenv("MIN_TRACK_FRAMES",   "3"))
AGENT_B_HOST     = os.getenv("AGENT_B_HOST", "192.168.1.100")
AGENT_B_PORT     = int(os.getenv("AGENT_B_PORT",       "9000"))
UAV_ID           = os.getenv("UAV_ID",   "UAV_1")
SECTOR           = os.getenv("SECTOR",   "DEMO")
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEVICE           = os.getenv("DEVICE",   "cpu")

OUTPUT = ROOT / "tests" / "output"

# ── colours (BGR) ─────────────────────────────────────────────────────────────
C_GREEN   = (0,   220,  60)
C_YELLOW  = (0,   200, 255)
C_RED     = (0,    40, 220)
C_BLUE    = (220, 100,   0)
C_WHITE   = (255, 255, 255)
C_BLACK   = (0,     0,   0)
C_CYAN    = (255, 220,   0)
C_ORANGE  = (0,   140, 255)
C_PANEL   = (30,   30,  30)   # status panel background

PANEL_W   = 420               # width of the right-hand status panel


# ═══════════════════════════════════════════════════════════════════════════════
# Shared state (written by caption thread, read by display loop)
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineState:
    def __init__(self):
        self.lock = threading.Lock()

        # LLaVA status
        self.llava_status   = "idle"      # "idle" | "running" | "done" | "error"
        self.llava_track_id = None
        self.llava_caption  = ""

        # MCP log — last N messages for the on-screen panel
        self.mcp_log: list = []           # list of dicts
        self.mcp_count = 0

        # File handle for persistent JSONL output (set by main before threads start)
        self.mcp_log_file = None          # open file object, written in add_mcp

        # TCP connection status (polled from sender)
        self.tcp_connected = False

        # DataFusion snapshot (updated each frame)
        self.n_active_tracks  = 0
        self.n_stable_tracks  = 0
        self.n_frames         = 0

        # Flash overlay when MCP fires
        self.flash_until = 0.0            # epoch time

    # helpers for thread-safe updates
    def set_llava(self, status, track_id=None, caption=""):
        with self.lock:
            self.llava_status   = status
            self.llava_track_id = track_id
            self.llava_caption  = caption

    def add_mcp(self, msg: dict):
        with self.lock:
            # Update on-screen panel log
            self.mcp_log.append(msg)
            if len(self.mcp_log) > 5:
                self.mcp_log.pop(0)
            self.mcp_count += 1
            self.flash_until = time.time() + 3.0   # flash for 3 s

            # Persist to JSONL file immediately so nothing is lost on exit
            if self.mcp_log_file:
                self.mcp_log_file.write(json.dumps(msg) + "\n")
                self.mcp_log_file.flush()


state = PipelineState()

# Queue: main loop → caption thread
# Each item: (track_id, frame_copy, best_det)
caption_queue: queue.Queue = queue.Queue(maxsize=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Caption worker thread
# ═══════════════════════════════════════════════════════════════════════════════

def caption_worker(captioner, fusion, sender):
    """
    Background thread that runs LLaVA and sends MCP once a caption is ready.
    Processes one request at a time so the video loop never blocks.
    """
    while True:
        try:
            track_id, frame, best_det, track = caption_queue.get(timeout=1)
        except queue.Empty:
            continue

        state.set_llava("running", track_id)
        logger.info("LLaVA captioning track %s …", track_id)

        try:
            caption = captioner.caption_detection(
                image=frame,
                detection_bbox=best_det.bbox_pixels,
                class_name=best_det.class_name,
                confidence=best_det.confidence,
            )
        except Exception as exc:
            logger.error("Caption exception: %s", exc)
            state.set_llava("error", track_id, str(exc))
            caption_queue.task_done()
            continue

        if not caption or caption.startswith("Error:"):
            logger.warning("Caption invalid: %s", caption)
            state.set_llava("error", track_id, caption or "empty")
            caption_queue.task_done()
            continue

        state.set_llava("done", track_id, caption)
        best_det.enrich(caption=caption)
        logger.info("Caption: %s", caption)

        # Build and send MCP pair
        msgs = fusion.create_mcp_messages(
            tracks=[track],
            uav_id=UAV_ID,
            sector=SECTOR,
            include_captions=True,
        )
        for msg in msgs:
            sent = sender.send(msg)
            logger.info("MCP %s %s", msg.get("type"), "SENT" if sent else "NO-CONN")
            state.add_mcp(msg)

        caption_queue.task_done()


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _text(img, txt, x, y, color=C_WHITE, scale=0.55, thickness=1):
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _wrap(text: str, max_chars: int) -> list:
    """Naive word-wrap to max_chars per line."""
    words, lines, current = text.split(), [], ""
    for w in words:
        if len(current) + len(w) + 1 > max_chars:
            if current:
                lines.append(current)
            current = w
        else:
            current = (current + " " + w).strip()
    if current:
        lines.append(current)
    return lines or [""]


def build_status_panel(h: int) -> np.ndarray:
    """Render the right-hand status panel as a (h × PANEL_W × 3) image."""
    panel = np.full((h, PANEL_W, 3), C_PANEL, dtype=np.uint8)
    x, y, dy = 14, 30, 26

    # ── Title ────────────────────────────────────────────────────────────────
    _text(panel, "VITALS VISION EDGE", x, y, C_CYAN, 0.65, 2);  y += dy + 4
    _text(panel, "Live Pipeline Demo", x, y, (160, 160, 160), 0.45);  y += dy + 8
    cv2.line(panel, (x, y), (PANEL_W - x, y), (80, 80, 80), 1);  y += 16

    # ── TCP connection ────────────────────────────────────────────────────────
    tcp_ok = state.tcp_connected
    tcp_col = C_GREEN if tcp_ok else C_RED
    _text(panel, "TCP Agent B", x, y, C_WHITE, 0.5);
    _text(panel, "CONNECTED" if tcp_ok else "WAITING…",
          PANEL_W - 120, y, tcp_col, 0.5, 2);  y += dy
    _text(panel, f"  {AGENT_B_HOST}:{AGENT_B_PORT}", x, y, (130, 130, 130), 0.42);  y += dy + 6
    cv2.line(panel, (x, y), (PANEL_W - x, y), (60, 60, 60), 1);  y += 14

    # ── Thresholds ───────────────────────────────────────────────────────────
    _text(panel, "Thresholds", x, y, C_WHITE, 0.5, 1);  y += dy
    _text(panel, f"  YOLO detect     ≥ {YOLO_CONF:.0%}", x, y, (180, 180, 180), 0.44);  y += dy - 4
    _text(panel, f"  Person emit     ≥ {PERSON_THRESH:.0%}", x, y, (180, 180, 180), 0.44);  y += dy - 4
    _text(panel, f"  LLaVA trigger   ≥ {CAPTION_THRESH:.0%}", x, y, (180, 180, 180), 0.44);  y += dy - 4
    _text(panel, f"  Stable frames   ≥ {MIN_TRACK_FRAMES}", x, y, (180, 180, 180), 0.44);  y += dy + 6
    cv2.line(panel, (x, y), (PANEL_W - x, y), (60, 60, 60), 1);  y += 14

    # ── DataFusion ───────────────────────────────────────────────────────────
    _text(panel, "DataFusion Tracking", x, y, C_WHITE, 0.5);  y += dy
    _text(panel, f"  Frame        #{state.n_frames}", x, y, (180, 180, 180), 0.44);  y += dy - 4
    _text(panel, f"  Active tracks  {state.n_active_tracks}", x, y, (180, 180, 180), 0.44);  y += dy - 4
    _text(panel, f"  Stable tracks  {state.n_stable_tracks}",
          x, y, C_GREEN if state.n_stable_tracks > 0 else (180, 180, 180), 0.44);  y += dy + 6
    cv2.line(panel, (x, y), (PANEL_W - x, y), (60, 60, 60), 1);  y += 14

    # ── LLaVA ────────────────────────────────────────────────────────────────
    llava_colours = {"idle": (130, 130, 130), "running": C_YELLOW,
                     "done": C_GREEN, "error": C_RED}
    llava_col = llava_colours.get(state.llava_status, C_WHITE)
    _text(panel, "LLaVA Captioner", x, y, C_WHITE, 0.5);  y += dy
    _text(panel, f"  Status:  {state.llava_status.upper()}", x, y, llava_col, 0.48, 2);  y += dy
    if state.llava_track_id:
        _text(panel, f"  Track:   {state.llava_track_id}", x, y, (160, 160, 160), 0.40);  y += dy - 4
    if state.llava_caption and state.llava_status == "done":
        for line in _wrap(state.llava_caption, 46)[:4]:
            _text(panel, f"  {line}", x, y, (200, 220, 200), 0.38);  y += 20
    y += 6
    cv2.line(panel, (x, y), (PANEL_W - x, y), (60, 60, 60), 1);  y += 14

    # ── MCP log ──────────────────────────────────────────────────────────────
    _text(panel, f"MCP Messages Sent  ({state.mcp_count} total)", x, y, C_WHITE, 0.5);  y += dy
    with state.lock:
        recent = list(state.mcp_log)
    if not recent:
        _text(panel, "  (none yet)", x, y, (100, 100, 100), 0.42);  y += dy
    else:
        for msg in reversed(recent[-3:]):
            msg_type = msg.get("type", "?").replace("MCP.", "")
            eid      = msg.get("event_id", "?")[:16]
            payload  = msg.get("payload", {})
            label    = payload.get("label", "?")
            conf     = payload.get("confidence", 0)
            col      = C_GREEN if msg_type == "Detection" else C_CYAN
            _text(panel, f"  [{msg_type}]", x, y, col, 0.42, 2);  y += 18
            _text(panel, f"    {label} {conf:.0%}  {eid}", x, y, (160, 200, 160), 0.38);  y += 20
    y += 4
    cv2.line(panel, (x, y), (PANEL_W - x, y), (60, 60, 60), 1);  y += 10

    # ── Controls ─────────────────────────────────────────────────────────────
    ctrl_y = h - 80
    _text(panel, "Controls", x, ctrl_y, (120, 120, 120), 0.44);  ctrl_y += 22
    _text(panel, "  Q  quit     R  reset fusion", x, ctrl_y, (100, 100, 100), 0.40);  ctrl_y += 18
    _text(panel, "  S  save frame", x, ctrl_y, (100, 100, 100), 0.40)

    return panel


def draw_pipeline_overlay(frame: np.ndarray, detections, fusion_stats: dict) -> np.ndarray:
    """Draw bounding boxes + per-detection label + frame-level overlay on the camera frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        x0, y0, x1, y1 = det.bbox_pixels
        is_person = det.class_name.lower() == "person"
        col = C_GREEN if (is_person and det.confidence >= PERSON_THRESH) else C_YELLOW

        # box
        cv2.rectangle(out, (x0, y0), (x1, y1), col, 2)

        # label background + text
        label = f"{det.class_name}  {det.confidence:.0%}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(out, (x0, y0 - th - bl - 6), (x0 + tw + 6, y0), col, -1)
        cv2.putText(out, label, (x0 + 3, y0 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_BLACK, 1, cv2.LINE_AA)

    # top-left frame info
    _text(out, f"Frame {fusion_stats['frame_number']:05d}", 10, 28, C_WHITE, 0.55, 1)
    _text(out, f"Detections: {len(detections)}", 10, 54, C_WHITE, 0.50, 1)

    # MCP flash overlay
    if time.time() < state.flash_until:
        alpha = min(1.0, (state.flash_until - time.time()) / 1.0)
        overlay = out.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 80, 0), -1)
        cv2.addWeighted(overlay, alpha * 0.6, out, 1 - alpha * 0.6, 0, out)
        _text(out, f"  MCP SENT  ({state.mcp_count} total)", 10, h - 20, C_GREEN, 0.65, 2)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(args):
    global AGENT_B_HOST, AGENT_B_PORT, CAMERA_INDEX, DEVICE

    if args.camera is not None:
        CAMERA_INDEX = args.camera
    if args.agent_b:
        host, _, port = args.agent_b.partition(":")
        AGENT_B_HOST = host
        if port:
            AGENT_B_PORT = int(port)

    print("\n" + "="*60)
    print("  VITALS Vision Edge — Live Pipeline Demo")
    print("="*60)
    print(f"  Camera      : index {CAMERA_INDEX}")
    print(f"  YOLO model  : {YOLO_MODEL}")
    print(f"  Ollama      : {OLLAMA_HOST}")
    print(f"  Agent B TCP : {AGENT_B_HOST}:{AGENT_B_PORT}")
    print(f"  Device      : {DEVICE}")
    print("="*60 + "\n")

    # ── Init modules ─────────────────────────────────────────────────────────
    print("Loading pipeline modules…")
    engine = PerceptionEngine(
        yolo_model_path=YOLO_MODEL,
        yolo_confidence=YOLO_CONF,
        caption_threshold=CAPTION_THRESH,
        device=DEVICE,
        ollama_host=OLLAMA_HOST,
    )
    fusion = DataFusion(
        iou_threshold=0.3,
        min_track_frames=MIN_TRACK_FRAMES,
        max_track_age=30,
    )
    sender = TCPSender(host=AGENT_B_HOST, port=AGENT_B_PORT)

    # Caption worker thread
    cap_thread = threading.Thread(
        target=caption_worker,
        args=(engine.captioner, fusion, sender),
        daemon=True,
    )
    cap_thread.start()

    # ── Camera ───────────────────────────────────────────────────────────────
    print(f"Opening camera {CAMERA_INDEX}…")
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera ready: {cam_w}×{cam_h}")

    # Tracks that have already been queued/captioned
    captioned_tracks: set = set()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Open the MCP log file for this session before the caption thread starts
    session_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    mcp_log_path = OUTPUT / f"mcp_log_{session_ts}.jsonl"
    state.mcp_log_file = open(mcp_log_path, "w", encoding="utf-8")
    print(f"MCP detections will be logged to: {mcp_log_path}")

    win_name = "VITALS Vision Edge — Demo  [Q=quit  R=reset  S=save]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cam_w + PANEL_W, cam_h)

    print("\nPipeline running.  Press Q in the window to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.02)
                continue

            # 1. YOLO
            detections, _ = engine.process_image(
                frame, drone_id=1, generate_captions=False, return_annotated=False,
            )

            # 2. DataFusion
            fusion.process_detections(detections, drone_id=1)
            f_stats = fusion.get_statistics()

            # 3. Update shared state for display
            state.tcp_connected    = sender.is_connected()
            state.n_frames         = f_stats["frame_number"]
            state.n_active_tracks  = f_stats["active_tracks"]
            state.n_stable_tracks  = f_stats["stable_tracks"]

            # 4. Check stable person tracks — queue for captioning if new
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
                # Offer to caption thread (non-blocking; skip if already busy)
                try:
                    caption_queue.put_nowait(
                        (track.track_id, frame.copy(), best_det, track)
                    )
                    captioned_tracks.add(track.track_id)
                    logger.info(
                        "Queued track %s for captioning (conf=%.2f, frames=%d)",
                        track.track_id, track.confidence, track.frame_count,
                    )
                except queue.Full:
                    pass   # caption thread is busy — will retry next frame

            # 5. Build display
            annotated  = draw_pipeline_overlay(frame, detections, f_stats)
            panel      = build_status_panel(cam_h)
            display    = np.hstack([annotated, panel])
            cv2.imshow(win_name, display)

            # 6. Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            elif key == ord("r"):
                fusion.reset()
                captioned_tracks.clear()
                with state.lock:
                    state.mcp_log.clear()
                    state.n_active_tracks = 0
                    state.n_stable_tracks = 0
                    state.n_frames = 0
                state.set_llava("idle")
                print("DataFusion state reset")
            elif key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = OUTPUT / f"demo_snapshot_{ts}.jpg"
                cv2.imwrite(str(path), display)
                print(f"Snapshot saved → {path}")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        sender.close()
        cv2.destroyAllWindows()
        if state.mcp_log_file:
            state.mcp_log_file.close()
        print(f"\nDemo ended.  {state.mcp_count} MCP message(s) logged to:")
        print(f"  {mcp_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VITALS Vision Edge — Live Demo")
    parser.add_argument("--camera",  type=int,  default=None,
                        help="Camera index (default: CAMERA_INDEX env or 0)")
    parser.add_argument("--agent-b", type=str,  default=None,
                        help="Agent B address, e.g. 192.168.1.100:9000")
    parser.add_argument("--device",  type=str,  default=None,
                        help="Inference device: cuda or cpu")
    args = parser.parse_args()
    if args.device:
        DEVICE = args.device
    main(args)
