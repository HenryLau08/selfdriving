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
from lidar_pipeline import (
    LidarReader, LidarVisualiser,
    bbox_angle_range, annotate_distance,
    distance_color_bgr, distance_label,
    DIST_DANGER, DIST_WARNING, IMAGE_WIDTH_PX,
    LIDAR_FORWARD_DEG,
)

import video as video

can = importlib.import_module("can")

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH     = "best.pt"   # or "best.pt" for CPU fallback
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
DETECTION_ID              = 0x440   # detections (class bitfield + speed)
LIDAR_ID                  = 0x450   # LIDAR distances per detection

# LIDAR
LIDAR_PORT      = "/dev/ttyUSB0"   # adjust if needed
LIDAR_ENABLED   = True            # set False to run without LIDAR
LIDAR_SHOW_VIS  = True            # show polar scan window

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


# ── OpenCV GUI availability check ────────────────────────────────────────────

def cv2_gui_available() -> bool:
    """Check whether OpenCV GUI functions are available (GTK/Qt/Cocoa backend)."""
    try:
        cv2.namedWindow("opencv_gui_test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("opencv_gui_test")
        return True
    except cv2.error:
        return False


def create_lidar_settings_window(initial_forward: int,
                                 initial_width: int) -> str:
    win_name = "LIDAR Settings"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 420, 80)
    cv2.createTrackbar("Forward", win_name, initial_forward, 360, lambda x: None)
    cv2.createTrackbar("Width", win_name, initial_width, 180, lambda x: None)
    return win_name


def get_lidar_settings(win_name: str) -> tuple[int, int]:
    forward = cv2.getTrackbarPos("Forward", win_name) % 360
    width   = cv2.getTrackbarPos("Width", win_name)
    return forward, max(10, width)


# ── Camera — same init as lane_follow ─────────────────────────────────────────

def initialize_camera() -> cv2.VideoCapture:
    config_dir = Path(__file__).resolve().parent / "configs"
    config     = video.get_camera_config(str(config_dir))

    if not config:
        print("No valid video configuration found!", file=sys.stderr)
        raise SystemExit(1)

    front = config.get("front")
    # if not front:
    #     print("Front camera not configured!", file=sys.stderr)
    #     raise SystemExit(1)

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
    return can.Bus(interface="socketcan", channel="vcan0", bitrate=500000)


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


# ── LIDAR CAN helpers ────────────────────────────────────────────────────────

def _make_lidar_message() -> Any:
    return can.Message(
        arbitration_id=LIDAR_ID,
        data=[0xFF] * 8,   # 0xFF = no reading
        is_extended_id=False,
    )


def _update_lidar_message(task: Any, message: Any,
                           forward_dist_m: float | None,
                           min_dist_m: float | None) -> None:
    """
    byte 0-1 : forward distance in cm (uint16, 0xFFFF = no reading)
    byte 2-3 : minimum distance across all detections in cm
    byte 4   : obstacle status  0=clear 1=warning 2=danger
    bytes 5-7: reserved
    """
    def to_cm(v): return min(0xFFFE, int(v * 100)) if v is not None else 0xFFFF
    fwd_cm = to_cm(forward_dist_m)
    min_cm = to_cm(min_dist_m)
    if min_dist_m is None:
        status = 0
    elif min_dist_m < DIST_DANGER:
        status = 2
    elif min_dist_m < DIST_WARNING:
        status = 1
    else:
        status = 0
    message.data = [
        (fwd_cm >> 8) & 0xFF, fwd_cm & 0xFF,
        (min_cm >> 8) & 0xFF, min_cm & 0xFF,
        status,
        0, 0, 0,
    ]
    task.modify_data(message)


# ── OCR helpers ─────────────────────────────────────────────────────────────────

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

def draw_box(frame, label, conf, speed, x1, y1, x2, y2):
    color = COLORS.get(label, (200, 200, 200))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{speed} km/h" if (label == "speed_sign" and speed) else f"{label} {conf:.2f}"
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

def run(device: str, use_can: bool, lidar_enabled: bool, lidar_port: str) -> None:
    print("Loading YOLO model …")
    model = YOLO(MODEL_PATH)
    print(f"  device : {device}")
    print(f"  classes: {list(model.names.values())}")

    print("Loading EasyOCR …")
    reader = easyocr.Reader(["en"], gpu=False)
    print("  OCR ready.")

    # ── LIDAR init ────────────────────────────────────────────────────────
    lidar     = None
    lidar_vis = None
    if lidar_enabled:
        print("Starting LIDAR …")
        lidar = LidarReader(port=lidar_port)
        if lidar.start():
            print("  LIDAR ready.")
            if LIDAR_SHOW_VIS:
                lidar_vis = LidarVisualiser()
        else:
            print("  LIDAR not available — continuing without it.")
            lidar = None

    bus = det_task = det_message = None
    lidar_task = lidar_message = None
    if use_can:
        bus         = initialize_can()
        det_message = _make_detection_message()
        det_task    = bus.send_periodic(det_message, CAN_MESSAGE_SENDING_SPEED)
        print("CAN bus ready — broadcasting on ID 0x440.")
        lidar_message = _make_lidar_message()
        lidar_task    = bus.send_periodic(lidar_message, CAN_MESSAGE_SENDING_SPEED)
        print("LIDAR CAN ready — broadcasting on ID 0x450.")

    camera = initialize_camera()
    print("Camera opened.")

    # Check OpenCV GUI availability
    gui_available = cv2_gui_available()
    if not gui_available:
        print("  ⚠  OpenCV GUI backend unavailable. Running without visualisation.")

    # Windows (only create if GUI is available)
    settings_window = None
    lidar_forward_deg = int(LIDAR_FORWARD_DEG) % 360
    lidar_front_width = 140
    if gui_available:
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Speed Sign OCR",   cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Speed Sign OCR",  320, 320)
        cv2.namedWindow("Traffic Light", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Traffic Light", 320, 320)
        settings_window = create_lidar_settings_window(lidar_forward_deg,
                                                       lidar_front_width)

    fps_times      = deque(maxlen=60)
    class_counter  = Counter()
    frame_idx      = 0
    last_speed     = None
    last_ocr_raw   = ""
    last_crop      = None
    last_tl_crop   = None
    last_tl_state  = "off"

    ocr_blank = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.putText(ocr_blank, "No speed sign yet", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    tl_blank = np.zeros((320, 320, 3), dtype=np.uint8)
    cv2.putText(tl_blank, "No traffic light yet", (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    if gui_available:
        cv2.imshow("Speed Sign OCR", ocr_blank)
        cv2.imshow("Traffic Light", tl_blank)

    print("\nRunning — press Q to quit.\n")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to read frame.")
                break

            if gui_available and settings_window is not None:
                lidar_forward_deg, lidar_front_width = get_lidar_settings(settings_window)

            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w         = frame_rgb.shape[:2]
            yolo_results = model.predict(frame_rgb, conf=CONFIDENCE,
                                         device=device, verbose=False)[0]

            detections   = []
            frame_speed  = None
            all_distances = []   # distances for all detections this frame

            for box in yolo_results.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                label  = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # ── LIDAR distance for this bounding box ─────────────────
                dist_m = None
                if lidar is not None:
                    a_min, a_max = bbox_angle_range(x1, x2, w,
                                                    forward_deg=lidar_forward_deg)
                    dist_m = lidar.get_distance_in_sector(a_min, a_max)
                    if dist_m is not None:
                        all_distances.append(dist_m)

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
                        if gui_available:
                            cv2.imshow("Speed Sign OCR", ocr_disp)

                    speed = last_speed
                    if speed:
                        frame_speed = speed

                if cls_id == TRAFFIC_LIGHT_ID:
                    last_tl_crop = frame_rgb[y1:y2, x1:x2]
                    last_tl_state = detect_traffic_light_color(last_tl_crop)

                detections.append({
                    "label":      label,
                    "class_id":   cls_id,
                    "confidence": round(conf, 3),
                    "bbox":       (x1, y1, x2, y2),
                    "speed_kmh":  speed,
                    "dist_m":     dist_m,
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

            # ── LIDAR CAN update ─────────────────────────────────────────────
            if use_can and lidar_task is not None:
                half = lidar_front_width / 2.0
                a_min = (lidar_forward_deg - half) % 360
                a_max = (lidar_forward_deg + half) % 360
                fwd_dist   = lidar.get_distance_in_sector(a_min, a_max) if lidar else None
                min_dist   = min(all_distances) if all_distances else None
                _update_lidar_message(lidar_task, lidar_message, fwd_dist, min_dist)

            # ── Draw & display ───────────────────────────────────────────────
            display = frame.copy()
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                draw_box(display, d["label"], d["confidence"],
                         d["speed_kmh"], x1, y1, x2, y2)
                annotate_distance(display, x1, y1, x2, y2, d.get("dist_m"))

            fps_times.append(time.perf_counter())
            fps = (len(fps_times) - 1) / (
                fps_times[-1] - fps_times[0] + 1e-6) if len(fps_times) > 1 else 0
            draw_stats(display, fps, device.upper(),
                       frame_speed or last_speed, class_counter)

            if gui_available:
                cv2.putText(display,
                            f"LIDAR forward={lidar_forward_deg}° width={lidar_front_width}°",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (220, 220, 220), 1)
                cv2.imshow("Object Detection", display)

                if last_tl_crop is not None:
                    tl_disp = draw_traffic_light_overlay(last_tl_crop, last_tl_state)
                else:
                    tl_disp = np.zeros((320, 320, 3), dtype=np.uint8)
                    cv2.putText(tl_disp, "No traffic light yet", (10, 160),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
                cv2.imshow("Traffic Light", tl_disp)

            # Update LIDAR polar plot
            if lidar_vis is not None:
                lidar_vis.update(lidar.get_full_scan(), img_width=w,
                                 forward_deg=lidar_forward_deg,
                                 width_deg=lidar_front_width)

            frame_idx += 1
            if gui_available and cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit.")
                break

    except KeyboardInterrupt:
        print("\nStopping …")

    finally:
        print("Shutting down …")
        if lidar:
            lidar.stop()
        if det_task:
            _update_detection_message(det_task, det_message, 0, 0, 0)
            time.sleep(0.1)
            det_task.stop()
        camera.release()
        if gui_available:
            cv2.destroyAllWindows()
        print("Done.")


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--device",  default="intel:npu",
#                         help="Inference device (default: intel:npu)")
#     parser.add_argument("--no-can",   action="store_true",
#                         help="Disable CAN bus (display only)")
#     parser.add_argument("--no-lidar", action="store_true",
#                         help="Disable LIDAR")
#     parser.add_argument("--lidar-port", default=LIDAR_PORT,
#                         help=f"LIDAR serial port (default: {LIDAR_PORT})")
#     args = parser.parse_args()
#     LIDAR_ENABLED = not args.no_lidar
#     LIDAR_PORT    = args.lidar_port
#     run(device=args.device, use_can=not args.no_can)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="intel:npu")
    parser.add_argument("--no-can", action="store_true")
    parser.add_argument("--no-lidar", action="store_true")
    parser.add_argument("--lidar-port", default=LIDAR_PORT)

    args = parser.parse_args()

    run(
        device=args.device,
        use_can=not args.no_can,
        lidar_enabled=not args.no_lidar,
        lidar_port=args.lidar_port,
    )

if __name__ == "__main__":
    main()
