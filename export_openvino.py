"""
Export trained YOLO .pt → OpenVINO format for NPU acceleration.
Run this once after training, before running detection_pipeline.py.

Usage:
    python export_openvino.py --model runs/detect/D_rob_coco_tt100k/weights/best.pt
    python export_openvino.py --model best.pt --int8   # INT8 quantisation (fastest)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import openvino as ov


def export(model_path: str, use_int8: bool, imgsz: int):
    model    = YOLO(model_path)
    out_path = Path(model_path).with_suffix("").parent / (
        Path(model_path).stem + "_openvino_model")

    print(f"Exporting: {model_path}")
    print(f"Format   : OpenVINO {'INT8' if use_int8 else 'FP16'}")
    print(f"imgsz    : {imgsz}")

    model.export(
        format="openvino",
        imgsz=imgsz,
        half=not use_int8,   # FP16 unless INT8 requested
        int8=use_int8,
    )

    print(f"\n✅ Export complete → {out_path}")

    # ── Verify NPU can load it ───────────────────────────────────────────────
    core      = ov.Core()
    available = core.available_devices
    print(f"Available OpenVINO devices: {available}")

    for device in ["NPU", "GPU", "CPU"]:
        if device in available:
            print(f"\nVerifying model loads on {device} …")
            try:
                ov_model  = core.read_model(str(out_path / "best.xml"))
                compiled  = core.compile_model(ov_model, device)
                print(f"✅ Model verified on {device}")
            except Exception as e:
                print(f"⚠ Could not load on {device}: {e}")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="Path to trained .pt file")
    parser.add_argument("--int8",  action="store_true",
                        help="Export as INT8 (fastest, slight accuracy drop)")
    parser.add_argument("--imgsz", default=640, type=int,
                        help="Input image size (must match training, default 640)")
    args = parser.parse_args()
    export(args.model, args.int8, args.imgsz)
