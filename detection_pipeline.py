"""
Object Detection + OCR Pipeline — integrated with NUC camera + CAN bus.
Runs alongside lane following; reads speed_sign via OCR and sends
the detected speed limit over CAN for the vehicle controller to use.

CAN message layout (DETECTION_ID = 0x440):
  byte 0     : number of detections this frame
  byte 1     : detected speed limit in km/h (0 = none)
  byte 2     : bitfield of present classes (bit i = class i detected)
  bytes 3–7  : reserved / zero

Usage (on NUC):
    python detection_pipeline.py
    python detection_pipeline.py --device intel:npu
    python detection_pipeline.py --no-can       # run without CAN (display only)
"""

import argparse
import importlib
import re
import struct
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
from traffic_light_color import detect_traffic_light_color, draw_traffic_light_overlay

import video as video

can = importlib.import_module("can")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "best_openvino_model"   # or "best.pt" for CPU fallback
CONFIDENCE     = 0.35
PADDING        = 12
OCR_EVERY_N    = 3          # run OCR every N frames (OCR is slow on CPU)
VALID_SPEEDS   = {10, 20, 30}

SCHEMA = [
    "person", "car", "traffic_light", "stop_sign",
    "left_turn", "speed_sign", "zebra_crossing", "car_sign",
]
SPEED_SIGN_ID      = SCHEMA.index("speed_sign")
TRAFFIC_LIGHT_ID   = SCHEMA.index("traffic_light")

# CAN
CAN_MESSAGE_SENDING_SPEED = 0.04
DETECTION_ID              = 0x440   # new message ID for detections

# BGR colours per class
COLORS = {
    "person":         (0,   220,  0),
    "car":            (0,   200, 255),
    "traffic_light":  (0,   140, 255),
    "stop_sign":      (0,   0,   230),
    "left_turn":      (220,  0,  200),
    "speed_sign":     (255, 140,   0),
    "zebra_crossing": (180, 180, 180),
    "car_sign":       (80,   80, 255),
}


# ── Camera — same init as lane_follow ─────────────────────────────────────────

def initialize_camera() -> cv2.VideoCapture:
    config_dir = Path(__file__).resolve().parent / "configs"
    config     = video.get_camera_config(str(config_dir))

    if not config:
        print("No valid video configuration found!", file=sys.stderr)
        raise SystemExit(1)

    front = config.get("front")
    if not front:
        print("Front camera not configured!", file=sys.stderr)
        raise SystemExit(1)

    cap = cv2.VideoCapture(front)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  848)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    0)
    cap.set(cv2.CAP_PROP_FOCUS,        0)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS,          30)

    if not cap.isOpened():
        print(f"Failed to open camera: {front}", file=sys.stderr)
        raise SystemExit(1)

    return cap


# ── CAN ───────────────────────────────────────────────────────────────────────

def initialize_can() -> Any:
    return can.Bus(interface="socketcan", channel="can0", bitrate=500000)


def _make_detection_message() -> Any:
    return can.Message(
        arbitration_id=DETECTION_ID,
        data=[0] * 8,
        is_extended_id=False,
    )


def _update_detection_message(task: Any, message: Any,
                               n_detections: int,
                               speed_kmh: int,
                               class_bitfield: int) -> None:
    message.data = [
        min(n_detections, 255),
        min(speed_kmh, 255),
        class_bitfield & 0xFF,
        0, 0, 0, 0, 0,
    ]
    task.modify_data(message)


def _build_class_bitfield(detections: list) -> int:
    bits = 0
    for d in detections:
        bits |= (1 << d["class_id"])
    return bits


# ── OCR helpers ───────────────────────────────────────────────────────────────

def extract_speed(text: str):
    nums  = re.findall(r'\b(\d{2,3})\b', text)
    valid = [int(n) for n in nums if int(n) in VALID_SPEEDS]
    return valid[0] if valid else (int(nums[0]) if nums else None)


def preprocess_crop(crop_rgb: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    scale = max(1, 80 // max(gray.shape[0], 1))
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_box(frame, label, conf, speed, x1, y1, x2, y2, tl_color=None):
    color = COLORS.get(label, (200, 200, 200))
    if label == "traffic_light" and tl_color:
        color = {"red": (0, 0, 230), "green": (0, 200, 0), "off": (120, 120, 120)}.get(tl_color, color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label == "speed_sign" and speed:
        text = f"{speed} km/h"
    elif label == "traffic_light" and tl_color:
        text = f"light: {tl_color}"
    else:
        text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(y1 - th - 8, 0)),
                  (x1 + tw + 4, max(y1, th + 8)), color, -1)
    cv2.putText(frame, text, (x1 + 2, max(y1 - 4, th + 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_stats(frame, fps: float, device: str, speed_kmh, class_counter: Counter):
    lines = [
        f"FPS: {fps:.1f}  [{device}]",
        f"Speed limit: {speed_kmh if speed_kmh else '--'} km/h",
        "─" * 24,
    ]
    for cls in SCHEMA:
        n = class_counter.get(cls, 0)
        if n:
            lines.append(f"{cls:<18} {n:>4}")

    x0, y0, pad, lh = 8, 8, 6, 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0),
                  (x0 + 210, y0 + pad * 2 + lh * len(lines)),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        color = (0, 255, 180) if i < 2 else (220, 220, 220)
        cv2.putText(frame, line,
                    (x0 + pad, y0 + pad + lh * (i + 1) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(device: str, use_can: bool) -> None:
    print("Loading YOLO model …")
    model = YOLO(MODEL_PATH)
    print(f"  device : {device}")
    print(f"  classes: {list(model.names.values())}")

    print("Loading EasyOCR …")
    reader = easyocr.Reader(["en"], gpu=False)
    print("  OCR ready.")

    bus = det_task = det_message = None
    if use_can:
        bus         = initialize_can()
        det_message = _make_detection_message()
        det_task    = bus.send_periodic(det_message, CAN_MESSAGE_SENDING_SPEED)
        print("CAN bus ready — broadcasting on ID 0x440.")

    camera = initialize_camera()
    print("Camera opened.")

    # Windows
    cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Speed Sign OCR",   cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Speed Sign OCR",  320, 320)
    cv2.namedWindow("Traffic Light",      cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Light",     160, 160)

    fps_times     = deque(maxlen=60)
    class_counter = Counter()
    frame_idx     = 0
    last_speed    = None
    last_ocr_raw  = ""
    last_crop     = None
    last_tl_color = "off"

    blank = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.putText(blank, "No speed sign yet", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.imshow("Speed Sign OCR", blank)

    tl_blank = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.putText(tl_blank, "No light yet", (6, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.imshow("Traffic Light", tl_blank)

    print("\nRunning — press Q to quit.\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame.")
                break

            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w         = frame_rgb.shape[:2]
            yolo_results = model.predict(frame_rgb, conf=CONFIDENCE,
                                         device=device, verbose=False)[0]

            detections   = []
            frame_speed  = None

            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                speed = None
                if cls_id == SPEED_SIGN_ID:
                    if frame_idx % OCR_EVERY_N == 0:
                        x1p = max(0, x1 - PADDING);  x2p = min(w, x2 + PADDING)
                        y1p = max(0, y1 - PADDING);  y2p = min(h, y2 + PADDING)
                        crop         = frame_rgb[y1p:y2p, x1p:x2p]
                        proc         = preprocess_crop(crop)
                        ocr_out      = reader.readtext(proc, detail=1)
                        last_ocr_raw = " ".join([r[1] for r in ocr_out])
                        last_speed   = extract_speed(last_ocr_raw)
                        last_crop    = crop
                        print(f"[{frame_idx}] OCR: '{last_ocr_raw}' → {last_speed} km/h")

                        # Update OCR window
                        ocr_disp = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                        ocr_disp = cv2.resize(ocr_disp, (320, 320),
                                              interpolation=cv2.INTER_NEAREST)
                        lbl = f"{last_speed} km/h" if last_speed else f"'{last_ocr_raw}'"
                        cv2.putText(ocr_disp, lbl, (8, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 180), 2)
                        cv2.imshow("Speed Sign OCR", ocr_disp)

                    speed = last_speed
                    if speed:
                        frame_speed = speed

                # ── Traffic light color detection ────────────────────────────────
                tl_color = None
                if cls_id == TRAFFIC_LIGHT_ID:
                    x1p = max(0, x1 - PADDING);  x2p = min(w, x2 + PADDING)
                    y1p = max(0, y1 - PADDING);  y2p = min(h, y2 + PADDING)
                    tl_crop       = frame_rgb[y1p:y2p, x1p:x2p]
                    last_tl_color = detect_traffic_light_color(tl_crop)
                    tl_color      = last_tl_color
                    tl_disp = draw_traffic_light_overlay(tl_crop, last_tl_color)
                    cv2.imshow("Traffic Light", tl_disp)
                    print(f"[{frame_idx}] traffic_light → {last_tl_color}")

                detections.append({
                    "label": label, "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "bbox": (x1, y1, x2, y2),
                    "speed_kmh": speed,
                    "tl_color":  tl_color,
                })
                class_counter[label] += 1

            # ── CAN update ───────────────────────────────────────────────────
            if use_can and det_task is not None:
                _update_detection_message(
                    det_task, det_message,
                    n_detections=len(detections),
                    speed_kmh=frame_speed or 0,
                    class_bitfield=_build_class_bitfield(detections),
                )

            # ── Draw & display ───────────────────────────────────────────────
            display = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                draw_box(display, d["label"], d["confidence"],
                         d["speed_kmh"], x1, y1, x2, y2,
                         tl_color=d.get("tl_color"))

            fps_times.append(time.perf_counter())
            fps = (len(fps_times) - 1) / (
                fps_times[-1] - fps_times[0] + 1e-6) if len(fps_times) > 1 else 0
            draw_stats(display, fps, device.upper(),
                       frame_speed or last_speed, class_counter)

            cv2.imshow("Object Detection", display)

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit.")
                break

    except KeyboardInterrupt:
        print("\nStopping …")

    finally:
        print("Shutting down …")
        if det_task:
            _update_detection_message(det_task, det_message, 0, 0, 0)
            time.sleep(0.1)
            det_task.stop()
        camera.release()
        cv2.destroyAllWindows()
        print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="intel:npu",
                        help="Inference device (default: intel:npu)")
    parser.add_argument("--no-can", action="store_true",
                        help="Disable CAN bus (display only)")
    args = parser.parse_args()
    run(device=args.device, use_can=not args.no_can)


if __name__ == "__main__":
    main()
