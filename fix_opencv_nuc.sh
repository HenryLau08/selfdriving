#!/bin/bash
# ── Quick fix for existing installs where opencv-headless was already installed
# Run this inside your activated venv:
#   source venv/bin/activate
#   bash fix_opencv_nuc.sh

echo "Fixing OpenCV display support on NUC ..."

# Ensure GTK and Qt system libs are present
sudo apt install -y \
    libgtk-3-0 \
    libgtk2.0-0 \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5core5a \
    python3-opencv

# Swap headless → full opencv inside venv
pip uninstall -y opencv-python-headless 2>/dev/null && echo "Removed headless" || echo "headless not installed"
pip uninstall -y opencv-python           2>/dev/null && echo "Removed opencv-python" || true
pip install opencv-python

# Verify
echo ""
echo "OpenCV GUI backend check:"
python3 - << 'PYCHECK'
import cv2
build = cv2.getBuildInformation()
found = False
for line in build.splitlines():
    if any(k in line for k in ["GTK", "Qt", "GUI"]):
        print(" ", line.strip())
        found = True
if not found:
    print("  WARNING: No GUI backend found — check output above")
else:
    print("")
    print("If GTK or Qt shows YES above, display should work.")
    print("If both show NO, recreate venv with:")
    print("  python3 -m venv --system-site-packages venv")
PYCHECK
