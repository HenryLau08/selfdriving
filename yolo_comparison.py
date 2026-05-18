import cv2
import numpy as np
import os


# =========================================================
# CONFIG
# =========================================================

VIDEO_NANO = "runs/detect/predict-3/latest_26n.avi"
VIDEO_SMALL = "runs/detect/predict-4/latest_26s.avi"
VIDEO_MEDIUM = "runs/detect/predict-5/latest_26m.avi"

OUTPUT_PATH = "yolo_grid_comparison.mp4"

WINDOW_NAME = "YOLO 2x2 Comparison"


# =========================================================
# GRID SETTINGS
# =========================================================

W, H = 640, 360  # per video tile

GRID_W = W * 2
GRID_H = H * 2


# =========================================================
# OPEN VIDEOS
# =========================================================

caps = [
    cv2.VideoCapture(VIDEO_NANO),
    cv2.VideoCapture(VIDEO_SMALL),
    cv2.VideoCapture(VIDEO_MEDIUM),
]

for i, cap in enumerate(caps):
    if not cap.isOpened():
        raise ValueError(f"Cannot open video {i}")


# =========================================================
# OUTPUT VIDEO WRITER
# =========================================================

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"), # type: ignore
    30,
    (GRID_W, GRID_H)
)


# =========================================================
# LABEL FUNCTION
# =========================================================

def add_label(frame, text):
    cv2.putText(
        frame,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )
    return frame


# =========================================================
# EMPTY TILE
# =========================================================

def empty_tile():
    return np.zeros((H, W, 3), dtype=np.uint8)


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    frames = []
    active = False

    for cap in caps:

        ret, frame = cap.read()

        if not ret:
            frame = None
        else:
            active = True

        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)

        frame = cv2.resize(frame, (W, H))

        frames.append(frame)

    if not active:
        break

    # =====================================================
    # LABELS
    # =====================================================

    frames[0] = add_label(frames[0], "YOLO Nano")
    frames[1] = add_label(frames[1], "YOLO Small")
    frames[2] = add_label(frames[2], "YOLO Medium")

    # =====================================================
    # BUILD GRID
    # =====================================================

    top_row = np.hstack((frames[0], frames[1]))
    bottom_row = np.hstack((frames[2], empty_tile()))

    grid = np.vstack((top_row, bottom_row))

    # =====================================================
    # WRITE + SHOW
    # =====================================================

    out.write(grid)

    cv2.imshow(WINDOW_NAME, grid)

    if cv2.waitKey(1) == 27:
        break


# =========================================================
# CLEANUP
# =========================================================

for cap in caps:
    cap.release()

out.release()
cv2.destroyAllWindows()

print(f"Saved: {os.path.abspath(OUTPUT_PATH)}")