"""
LIDAR Pipeline — RPLidar integration for obstacle distance detection.
Runs as a background thread and provides per-angle distance readings
that can be queried by the detection pipeline to annotate bounding boxes
with real-world distances.

Fusion approach:
  - RPLidar gives a 360° horizontal scan (angle, distance_mm)
  - Camera bounding boxes give (x1, x2) pixel columns
  - We map pixel columns → LIDAR angles using camera horizontal FOV
  - For each bounding box we return the minimum distance in that angle sector
    (closest point = most relevant obstacle)

Requirements:
    pip install rplidar-roboticia

Usage (standalone test):
    python lidar_pipeline.py --port /dev/ttyUSB0
    python lidar_pipeline.py --port /dev/ttyUSB0 --show
"""

import argparse
import math
import threading
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    from rplidar import RPLidar, RPLidarException
    RPLIDAR_AVAILABLE = True
except ImportError:
    RPLIDAR_AVAILABLE = False
    print("⚠  rplidar not installed — run: pip install rplidar-roboticia")


# ── Config ────────────────────────────────────────────────────────────────────

LIDAR_PORT       = "/dev/ttyUSB0"   # change to /dev/ttyUSB1 etc. if needed
LIDAR_BAUDRATE   = 115200

# Camera field of view (degrees, horizontal).
# Common values: 60° (narrow), 78° (standard webcam), 90° (wide)
# Measure or look up your camera's spec sheet.
CAMERA_HFOV_DEG  = 78.0

# Image width in pixels (must match cap resolution)
IMAGE_WIDTH_PX   = 848

# Which LIDAR angles face forward (towards the camera's view).
# 0° = forward on the RPLidar.  Adjust if the LIDAR is mounted rotated.
# e.g. if LIDAR is mounted facing backwards, set LIDAR_FORWARD_DEG = 180
LIDAR_FORWARD_DEG = 0.0

# Distance thresholds (metres)
DIST_DANGER  = 0.5    # very close — stop / emergency brake
DIST_WARNING = 1.5    # close — slow down / prepare to change lane
DIST_CLEAR   = 3.0    # far enough — lane change safe

# Minimum valid distance (filter out noise / self-detections)
DIST_MIN_M   = 0.15
DIST_MAX_M   = 12.0

# How many full scans to keep in the rolling buffer
SCAN_BUFFER  = 3


# ── Angle ↔ Pixel mapping ─────────────────────────────────────────────────────

def pixel_to_lidar_angle(x_pixel: int, img_width: int = IMAGE_WIDTH_PX,
                          hfov_deg: float = CAMERA_HFOV_DEG,
                          forward_deg: float = LIDAR_FORWARD_DEG) -> float:
    """
    Convert a horizontal pixel position to the corresponding LIDAR angle.

    The camera centre maps to LIDAR_FORWARD_DEG.
    Left edge of image → forward_deg - hfov/2
    Right edge         → forward_deg + hfov/2
    """
    # Normalise to [-0.5, +0.5]
    norm  = (x_pixel / img_width) - 0.5
    angle = forward_deg + norm * hfov_deg
    return angle % 360.0


def bbox_angle_range(x1: int, x2: int,
                     img_width: int = IMAGE_WIDTH_PX,
                     hfov_deg: float = CAMERA_HFOV_DEG,
                     forward_deg: float = LIDAR_FORWARD_DEG
                     ) -> tuple[float, float]:
    """Return (angle_left, angle_right) for a bounding box column range."""
    a_left  = pixel_to_lidar_angle(x1, img_width, hfov_deg, forward_deg)
    a_right = pixel_to_lidar_angle(x2, img_width, hfov_deg, forward_deg)
    return a_left, a_right


def angles_in_range(angle: float, a_min: float, a_max: float) -> bool:
    """Check if angle falls within [a_min, a_max], handling 360° wrap."""
    angle = angle % 360
    a_min = a_min % 360
    a_max = a_max % 360
    if a_min <= a_max:
        return a_min <= angle <= a_max
    # wraps around 0°
    return angle >= a_min or angle <= a_max


# ── LIDAR thread ──────────────────────────────────────────────────────────────

class LidarReader:
    """
    Runs the RPLidar in a background thread.
    Maintains a rolling buffer of recent scans.
    Thread-safe distance queries via get_distance_in_sector().
    """

    def __init__(self, port: str = LIDAR_PORT,
                 baudrate: int = LIDAR_BAUDRATE):
        self.port      = port
        self.baudrate  = baudrate
        self._lock     = threading.Lock()
        self._scans    = deque(maxlen=SCAN_BUFFER)   # each scan: list of (angle, dist_m)
        self._running  = False
        self._thread   = None
        self._lidar    = None
        self.connected = False
        self.scan_count = 0

    def start(self) -> bool:
        """Start the background reader thread. Returns True if LIDAR connected."""
        if not RPLIDAR_AVAILABLE:
            print("  ⚠  rplidar library not available.")
            return False
        try:
            self._lidar = RPLidar(self.port, baudrate=self.baudrate)
            info = self._lidar.get_info()
            health = self._lidar.get_health()
            print(f"  RPLidar connected: {info}")
            print(f"  Health: {health}")
            self._running  = True
            self.connected = True
            self._thread   = threading.Thread(target=self._read_loop,
                                               daemon=True, name="LidarReader")
            self._thread.start()
            return True
        except Exception as e:
            print(f"  ⚠  Could not connect to RPLidar on {self.port}: {e}")
            self.connected = False
            return False

    def stop(self):
        """Stop the reader thread and close the LIDAR."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._lidar:
            try:
                self._lidar.stop()
                self._lidar.stop_motor()
                self._lidar.disconnect()
            except Exception:
                pass
        print("  LIDAR stopped.")

    def _read_loop(self):
        """Background thread: continuously read scans from RPLidar."""
        try:
            for scan in self._lidar.iter_scans(max_buf_meas=500):
                if not self._running:
                    break
                points = []
                for (_, angle, dist_mm) in scan:
                    dist_m = dist_mm / 1000.0
                    if DIST_MIN_M <= dist_m <= DIST_MAX_M:
                        points.append((float(angle), dist_m))
                if points:
                    with self._lock:
                        self._scans.append(points)
                        self.scan_count += 1
        except RPLidarException as e:
            if self._running:
                print(f"  LIDAR read error: {e}")
        except Exception as e:
            if self._running:
                print(f"  LIDAR unexpected error: {e}")

    def get_distance_in_sector(self, a_min: float, a_max: float) -> Optional[float]:
        """
        Return the minimum distance (metres) of any LIDAR point whose angle
        falls in [a_min, a_max].  Returns None if no points in sector.
        Uses the most recent SCAN_BUFFER scans for stability.
        """
        with self._lock:
            if not self._scans:
                return None
            distances = []
            for scan in self._scans:
                for (angle, dist_m) in scan:
                    if angles_in_range(angle, a_min, a_max):
                        distances.append(dist_m)
        return min(distances) if distances else None

    def get_forward_distance(self, sector_deg: float = 20.0) -> Optional[float]:
        """
        Convenience: minimum distance directly ahead
        (±sector_deg/2 around LIDAR_FORWARD_DEG).
        """
        half = sector_deg / 2
        a_min = (LIDAR_FORWARD_DEG - half) % 360
        a_max = (LIDAR_FORWARD_DEG + half) % 360
        return self.get_distance_in_sector(a_min, a_max)

    def get_full_scan(self) -> list:
        """Return a copy of the latest scan for visualisation."""
        with self._lock:
            if not self._scans:
                return []
            return list(self._scans[-1])


# ── Distance annotation ───────────────────────────────────────────────────────

def distance_label(dist_m: Optional[float]) -> str:
    """Human-readable distance string."""
    if dist_m is None:
        return ""
    return f"{dist_m:.2f}m"


def distance_color_bgr(dist_m: Optional[float]) -> tuple:
    """
    Traffic-light colour based on distance:
      red    → danger  (< DIST_DANGER)
      orange → warning (< DIST_WARNING)
      green  → clear   (>= DIST_CLEAR)
    """
    if dist_m is None:
        return (180, 180, 180)
    if dist_m < DIST_DANGER:
        return (0, 0, 255)      # red
    if dist_m < DIST_WARNING:
        return (0, 140, 255)    # orange
    return (0, 200, 0)          # green


def annotate_distance(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       dist_m: Optional[float]) -> None:
    """Draw distance label below the bounding box."""
    if dist_m is None:
        return
    label = distance_label(dist_m)
    color = distance_color_bgr(dist_m)
    cv2.putText(frame, label,
                (x1 + 2, y2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


# ── LIDAR visualisation window ────────────────────────────────────────────────

class LidarVisualiser:
    """
    Optional polar-plot window showing the live LIDAR scan.
    Red wedge = camera FOV. Dots coloured by distance.
    """
    WIN  = "LIDAR Scan"
    SIZE = 400      # pixels per side

    def __init__(self):
        import cv2
        cv2.namedWindow(self.WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WIN, self.SIZE, self.SIZE)
        self._blank()

    def _blank(self):
        import cv2
        canvas = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        cx, cy = self.SIZE // 2, self.SIZE // 2
        cv2.circle(canvas, (cx, cy), cx - 4, (30, 30, 30), 1)
        cv2.circle(canvas, (cx, cy), (cx - 4) // 2, (30, 30, 30), 1)
        cv2.imshow(self.WIN, canvas)

    def update(self, scan: list, img_width: int = IMAGE_WIDTH_PX):
        import cv2
        canvas = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)
        cx, cy = self.SIZE // 2, self.SIZE // 2
        scale  = (self.SIZE // 2 - 10) / DIST_MAX_M

        # Draw range rings
        for r_m in [1, 2, 3, 5]:
            r_px = int(r_m * scale)
            cv2.circle(canvas, (cx, cy), r_px, (40, 40, 40), 1)
            cv2.putText(canvas, f"{r_m}m", (cx + r_px + 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)

        # Draw camera FOV wedge
        fov_half = CAMERA_HFOV_DEG / 2
        a1 = math.radians(LIDAR_FORWARD_DEG - fov_half - 90)
        a2 = math.radians(LIDAR_FORWARD_DEG + fov_half - 90)
        pts = [(cx, cy)]
        for a in np.linspace(a1, a2, 30):
            r = self.SIZE // 2 - 5
            pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], (0, 20, 40))

        # Draw LIDAR points
        for (angle, dist_m) in scan:
            rad = math.radians(angle - 90)
            px  = int(cx + dist_m * scale * math.cos(rad))
            py  = int(cy + dist_m * scale * math.sin(rad))
            color = distance_color_bgr(dist_m)
            cv2.circle(canvas, (px, py), 2, color, -1)

        # Forward direction marker
        cv2.arrowedLine(canvas, (cx, cy),
                        (cx, cy - 30), (0, 255, 180), 1, tipLength=0.3)

        cv2.imshow(self.WIN, canvas)


# ── Standalone test ───────────────────────────────────────────────────────────

def main():
    import cv2

    parser = argparse.ArgumentParser(description="RPLidar standalone test")
    parser.add_argument("--port",  default=LIDAR_PORT)
    parser.add_argument("--show",  action="store_true",
                        help="Show polar visualisation window")
    args = parser.parse_args()

    lidar = LidarReader(port=args.port)
    if not lidar.start():
        print("Failed to start LIDAR. Check --port and permissions.")
        print("  Try: sudo chmod 666 /dev/ttyUSB0")
        return

    vis = LidarVisualiser() if args.show else None

    print("\nReading LIDAR — press Ctrl+C to stop\n")
    try:
        while True:
            fwd = lidar.get_forward_distance()
            scan = lidar.get_full_scan()

            print(f"  Scans: {lidar.scan_count:5d} | "
                  f"Points: {len(scan):4d} | "
                  f"Forward: {f'{fwd:.2f}m' if fwd else 'N/A':>8s} | "
                  f"Status: {distance_color_bgr(fwd)}", end="\r")

            if vis and scan:
                vis.update(scan)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopping …")
    finally:
        lidar.stop()
        if vis:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
