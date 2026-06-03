#!/bin/bash
# ── Linux install script (Ubuntu/Debian — NUC with display output)
# Run with: bash install_linux.sh

set -e

echo "======================================================"
echo " Self-driving detection pipeline — Linux install"
echo "======================================================"

# ── 1. System packages ─────────────────────────────────────────────────────
echo ""
echo "[1/5] Installing system packages via apt..."
sudo apt update -q
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-opencv \
    libopencv-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-0 \
    libgtk2.0-0 \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    v4l-utils \
    can-utils

# ── 2. Create venv WITH system site-packages ───────────────────────────────
# --system-site-packages lets the venv fall back to system opencv (which has
# GTK compiled in) if the pip version has display issues.
echo ""
echo "[2/5] Creating virtual environment (with system site-packages) ..."
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip wheel

# ── 3. PyTorch CPU ────────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch (CPU build) ..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# ── 4. Detection packages ─────────────────────────────────────────────────
# Use opencv-python (NOT headless) — headless has no imshow/GUI support.
# If already installed as headless, uninstall it first.
echo ""
echo "[4/5] Installing detection packages ..."
pip uninstall -y opencv-python-headless 2>/dev/null || true
pip install \
    ultralytics \
    easyocr \
    opencv-python \
    numpy \
    python-can

# ── 5. OpenVINO ───────────────────────────────────────────────────────────
echo ""
echo "[5/5] Installing OpenVINO runtime ..."
pip install openvino

# ── Verify OpenCV display backend ─────────────────────────────────────────
echo ""
echo "Verifying OpenCV build info ..."
python3 - << 'PYCHECK'
import cv2
build = cv2.getBuildInformation()
# Check which GUI backend is available
for line in build.splitlines():
    if any(k in line for k in ["GTK", "Qt", "WIN32", "COCOA", "GUI"]):
        print(" ", line.strip())
PYCHECK

echo ""
echo "======================================================"
echo " Install complete!"
echo ""
echo " Activate the environment before running:"
echo "   source venv/bin/activate"
echo ""
echo " Then run:"
echo "   python test_detection.py --model best.pt"
echo ""
echo " If you still see display errors, run:"
echo "   python3 -c \"import cv2; print(cv2.getBuildInformation())\" | grep -i gtk"
echo "======================================================"
