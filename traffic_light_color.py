# ── Traffic light color detection (shared module) ─────────────────────────────
# Imported by both detection_pipeline.py and test_detection.py

import cv2
import numpy as np


def detect_traffic_light_color(crop_rgb: np.ndarray) -> str:
    """
    Classify a traffic light crop as 'red', 'green', or 'off'.

    Strategy:
      1. Resize crop to a fixed height so analysis is scale-invariant
      2. Convert to HSV — more robust to lighting than raw RGB
      3. Build masks for red and green hue ranges
      4. Compare brightness-weighted pixel counts in top vs bottom half
         (red light is at top, green at bottom on a standard traffic light)
      5. If neither colour is bright enough → 'off'

    Returns: 'red' | 'green' | 'off'
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return "off"

    # Resize to fixed height for consistent analysis
    h, w = crop_rgb.shape[:2]
    target_h = 96
    scale    = target_h / max(h, 1)
    resized  = cv2.resize(crop_rgb, (max(1, int(w * scale)), target_h),
                          interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    v   = hsv[:, :, 2]   # value channel (brightness)

    # ── Colour masks ──────────────────────────────────────────────────────────
    # Red wraps around 0/180 in HSV
    red_mask = (
        cv2.inRange(hsv, np.array([0,  100, 100]), np.array([10,  255, 255])) |
        cv2.inRange(hsv, np.array([165, 100, 100]), np.array([180, 255, 255]))
    )
    green_mask = cv2.inRange(
        hsv, np.array([40, 60, 80]), np.array([90, 255, 255]))

    # ── Weighted score: mask × brightness ────────────────────────────────────
    def score(mask):
        return float(np.sum((mask > 0) * v.astype(float)))

    red_score   = score(red_mask)
    green_score = score(green_mask)

    # Minimum brightness threshold to avoid classifying a dark/off light
    MIN_SCORE = 500.0

    if red_score < MIN_SCORE and green_score < MIN_SCORE:
        return "off"

    if red_score >= green_score:
        return "red"
    return "green"


def draw_traffic_light_overlay(crop_rgb: np.ndarray, color: str) -> np.ndarray:
    """Return a 160×160 BGR display image with the color label overlaid."""
    disp = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    disp = cv2.resize(disp, (160, 160), interpolation=cv2.INTER_NEAREST)

    label_color = {
        "red":   (0,   0,   255),
        "green": (0,   200,  0),
        "off":   (120, 120, 120),
    }.get(color, (200, 200, 200))

    # Coloured dot indicator
    cv2.circle(disp, (140, 20), 12, label_color, -1)
    cv2.putText(disp, color.upper(), (6, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, label_color, 2)
    return disp
