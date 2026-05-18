import os
import glob
import random
import cv2
import numpy as np
import streamlit as st


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(layout="wide")

folder_path = "extracted_data/right"
n_images = 10
seed = 42

extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


# =========================================================
# LOAD IMAGES
# =========================================================

all_images = []
for ext in extensions:
    all_images.extend(glob.glob(os.path.join(folder_path, ext)))

if len(all_images) == 0:
    st.error("No images found in folder.")
    st.stop()

random.seed(seed)
selected_images = random.sample(all_images, min(n_images, len(all_images)))


# =========================================================
# FUNCTIONS
# =========================================================

def roi(img, top_w, top_h):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)

    top_y = int(h * top_h / 100)
    left_x = int(w * (100 - top_w) / 200)
    right_x = w - left_x

    poly = np.array([[
        (0, h),
        (w, h),
        (right_x, top_y),
        (left_x, top_y)
    ]], np.int32)

    if len(img.shape) == 3:
        cv2.fillPoly(mask, poly, (255, 255, 255)) # type: ignore
    else:
        cv2.fillPoly(mask, poly, 255) # type: ignore

    return cv2.bitwise_and(img, mask)


def bird_view(img):
    h, w = img.shape[:2]

    src = np.float32([
        [w * 0.45, h * 0.63],
        [w * 0.55, h * 0.63],
        [w * 0.90, h],
        [w * 0.10, h]
    ]) # type: ignore

    dst = np.float32([
        [w * 0.20, 0],
        [w * 0.80, 0],
        [w * 0.80, h],
        [w * 0.20, h]
    ]) # type: ignore

    M = cv2.getPerspectiveTransform(src, dst) # type: ignore
    return cv2.warpPerspective(img, M, (w, h))


def to_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.title("Controls")

img_idx = st.sidebar.slider("Image Index", 0, len(selected_images) - 1, 0)

white_v = st.sidebar.slider("White Min V", 0, 255, 200)
white_s = st.sidebar.slider("White Max S", 0, 255, 40)

canny_min = st.sidebar.slider("Canny Min", 0, 255, 50)
canny_max = st.sidebar.slider("Canny Max", 0, 255, 150)

roi_w = st.sidebar.slider("ROI Width %", 0, 100, 20)
roi_h = st.sidebar.slider("ROI Height %", 0, 100, 60)

grad_thresh = st.sidebar.slider("Gradient Percentile", 0, 100, 60)


# =========================================================
# LOAD IMAGE
# =========================================================

path = selected_images[img_idx]
img = cv2.imread(path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # type: ignore
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # type: ignore


# =========================================================
# PROCESSING PIPELINE
# =========================================================

# White mask
lower = np.array([0, 0, white_v])
upper = np.array([180, white_s, 255])
white = cv2.inRange(hsv, lower, upper)

# Blur + edges
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, canny_min, canny_max)

# Sobel
sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3) # type: ignore
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, 3) # type: ignore

sobel_x = np.uint8(np.absolute(sx))
sobel_y = np.uint8(np.absolute(sy))

# Gradient magnitude
mag = np.sqrt(sx**2 + sy**2)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore

# percentile threshold
thr = np.percentile(mag, grad_thresh)
mask = (mag > thr).astype(np.uint8) * 255
grad_filtered = cv2.bitwise_and(mag, mag, mask=mask)

# combinations
white_edges = cv2.bitwise_and(edges, white)

# ROI
roi_edges = roi(edges, roi_w, roi_h)

# bird view
bird = bird_view(img)


# =========================================================
# UI LAYOUT
# =========================================================

st.title("🚗 Lane Detection EDA Dashboard")

st.text(f"File: {os.path.basename(path)}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Original",
    "Color",
    "Edges",
    "Gradients",
    "Geometry"
])


# =========================================================
# TAB 1 - ORIGINAL
# =========================================================

with tab1:
    st.image(to_rgb(img), use_container_width=True)


# =========================================================
# TAB 2 - COLOR
# =========================================================

with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.image(gray, caption="Grayscale", use_container_width=True)

    with col2:
        st.image(white, caption="White Mask", use_container_width=True)


# =========================================================
# TAB 3 - EDGES
# =========================================================

with tab3:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(edges, caption="Canny", use_container_width=True)

    with col2:
        st.image(white_edges, caption="White + Edges", use_container_width=True)

    with col3:
        st.image(roi_edges, caption="ROI Edges", use_container_width=True)


# =========================================================
# TAB 4 - GRADIENTS
# =========================================================

with tab4:
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(sobel_x, caption="Sobel X", use_container_width=True) # type: ignore

    with col2:
        st.image(sobel_y, caption="Sobel Y", use_container_width=True) # type: ignore

    with col3:
        st.image(mag, caption="Gradient Magnitude", use_container_width=True)

    st.image(grad_filtered, caption="Gradient Filtered (thresholded)", use_container_width=True)


# =========================================================
# TAB 5 - GEOMETRY
# =========================================================

with tab5:
    col1, col2 = st.columns(2)

    with col1:
        st.image(bird, caption="Bird View", use_container_width=True)

    with col2:
        st.write("### ROI Settings Preview")
        st.write(f"Width: {roi_w}%")
        st.write(f"Height: {roi_h}%")


# =========================================================
# METRICS
# =========================================================

brightness = np.mean(gray)
contrast = np.std(gray)
edge_density = np.mean(edges > 0)
white_ratio = np.mean(white > 0)

st.sidebar.markdown("---")
st.sidebar.write("### Metrics")
st.sidebar.write(f"Brightness: {brightness:.2f}")
st.sidebar.write(f"Contrast: {contrast:.2f}")
st.sidebar.write(f"Edge density: {edge_density:.4f}")
st.sidebar.write(f"White ratio: {white_ratio:.4f}")