"""
Label checker for Roboflow YOLO format exports.
- Reads data.yaml for class names and split paths
- Finds label .txt files by swapping images/ → labels/ in the path
- Draws bounding boxes on images to visually verify labels
- Reports which classes are present / missing

Usage:
    python check_labels.py --dataset roboflow_export/
    python check_labels.py --dataset roboflow_export.zip
    python check_labels.py --dataset roboflow_export/ --split train
    python check_labels.py --dataset roboflow_export/ --class left_turn
    python check_labels.py --dataset roboflow_export/ --no-show
"""

import argparse
import random
import sys
import zipfile
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import yaml

# ── Expected schema ───────────────────────────────────────────────────────────
EXPECTED_CLASSES = [
    "car", "car_sign", "left_turn", "person",
    "speed_sign", "stop_sign", "traffic_light", "zebra_crossing",
]

CLASS_COLORS_BGR = {
    "car":            (0,   200, 255),
    "car_sign":       (80,   80, 255),
    "left_turn":      (220,   0, 200),
    "person":         (0,   220,   0),
    "speed_sign":     (255, 140,   0),
    "stop_sign":      (0,     0, 230),
    "traffic_light":  (0,   140, 255),
    "zebra_crossing": (180, 180, 180),
}
DEFAULT_COLOR = (200, 200, 200)


# ── Setup ─────────────────────────────────────────────────────────────────────

def unzip_if_needed(path: Path) -> Path:
    if path.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        print(f"Unzipping {path} → {tmp} …")
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        candidates = list(tmp.rglob("data.yaml"))
        if not candidates:
            sys.exit("No data.yaml found inside zip.")
        return candidates[0].parent
    return path


def load_yaml(root: Path) -> tuple[list[str], dict[str, Path]]:
    """
    Parse data.yaml.
    Resolves split image dirs relative to the yaml file location.
    Returns (class_names, {split: abs_image_dir}).
    """
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        sys.exit(f"data.yaml not found in {root}")

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # class names
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]

    # resolve split paths — Roboflow uses paths relative to data.yaml
    split_dirs = {}
    for split_key, yaml_key in [("train", "train"), ("valid", "valid"), ("test", "test")]:
        raw = cfg.get(yaml_key)
        if not raw:
            continue
        # Strip leading ../  and resolve from root
        clean = raw.lstrip("./").lstrip("../")
        # Try relative to root first, then one level up (Roboflow uses ../)
        for base in [root, root.parent]:
            candidate = (base / clean).resolve()
            if candidate.exists():
                split_dirs[split_key] = candidate
                break
        else:
            # Last resort: try as-is relative to root
            split_dirs[split_key] = (root / clean).resolve()

    return names, split_dirs


def label_dir_from_image_dir(img_dir: Path) -> Path:
    """
    Roboflow YOLO structure:
        split/images/  ←  img_dir
        split/labels/  ←  returned
    """
    return img_dir.parent / "labels" / img_dir.name \
        if img_dir.parent.name != "images" \
        else img_dir.parent.parent / "labels"


# ── Load ──────────────────────────────────────────────────────────────────────

def load_split(img_dir: Path, names: list[str]) -> list[dict]:
    """
    Returns list of:
        { img_path: Path, labels: [(cls_id, cls_name, xc, yc, w, h)] }
    """
    if not img_dir.exists():
        print(f"  ⚠  Image dir not found: {img_dir}")
        return []

    # Label dir: swap 'images' segment for 'labels'
    parts = img_dir.parts
    if "images" in parts:
        lbl_parts = tuple("labels" if p == "images" else p for p in parts)
        lbl_dir   = Path(*lbl_parts)
    else:
        lbl_dir = img_dir.parent / "labels"

    if not lbl_dir.exists():
        print(f"  ⚠  Label dir not found: {lbl_dir}")
        return []

    print(f"  images → {img_dir}")
    print(f"  labels → {lbl_dir}")

    samples = []
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        # Find matching image
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = img_dir / (lbl_file.stem + ext)
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        labels = []
        for line in lbl_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id   = int(parts[0])
            cls_name = names[cls_id] if cls_id < len(names) else f"unknown_{cls_id}"
            xc, yc, w, h = map(float, parts[1:])
            # Basic sanity check on coords
            if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < w <= 1 and 0 < h <= 1):
                continue
            labels.append((cls_id, cls_name, xc, yc, w, h))

        if labels:
            samples.append({"img_path": img_path, "labels": labels})

    return samples


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_boxes(img_bgr: np.ndarray, labels: list) -> np.ndarray:
    out  = img_bgr.copy()
    ih, iw = out.shape[:2]
    for (_, cls_name, xc, yc, w, h) in labels:
        x1 = max(0, int((xc - w / 2) * iw))
        y1 = max(0, int((yc - h / 2) * ih))
        x2 = min(iw, int((xc + w / 2) * iw))
        y2 = min(ih, int((yc + h / 2) * ih))
        color = CLASS_COLORS_BGR.get(cls_name, DEFAULT_COLOR)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(
            cls_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, cls_name, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return out


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(all_data: dict, names: list) -> tuple[list, Counter]:
    print("\n" + "=" * 62)
    print("  LABEL CHECK REPORT")
    print("=" * 62)

    active_splits = [s for s, v in all_data.items() if v]
    total_images  = sum(len(v) for v in all_data.values())
    print(f"\nTotal labelled images : {total_images}")
    print(f"Splits loaded         : {active_splits}\n")

    # Count labels per class per split
    split_counts  = {s: Counter() for s in active_splits}
    class_totals  = Counter()
    for split, samples in all_data.items():
        for s in samples:
            for (_, cname, *_) in s["labels"]:
                split_counts[split][cname] += 1
                class_totals[cname]        += 1

    # Table header
    header = f"  {'Class':<22}"
    for split in active_splits:
        header += f" {split:>8}"
    header += f" {'TOTAL':>8}"
    print(header)
    print("  " + "-" * (22 + 9 * len(active_splits) + 9))

    for cls in EXPECTED_CLASSES:
        row = f"  {cls:<22}"
        for split in active_splits:
            row += f" {split_counts[split].get(cls, 0):>8}"
        row += f" {class_totals.get(cls, 0):>8}"
        if class_totals.get(cls, 0) == 0:
            row += "  ← ❌ MISSING"
        print(row)

    # Totals row
    row = f"  {'TOTAL':<22}"
    for split in active_splits:
        row += f" {sum(split_counts[split].values()):>8}"
    row += f" {sum(class_totals.values()):>8}"
    print("  " + "-" * (22 + 9 * len(active_splits) + 9))
    print(row)

    # Class presence summary
    missing = [c for c in EXPECTED_CLASSES if class_totals.get(c, 0) == 0]
    print("\n── Class presence ───────────────────────────────────────")
    for cls in EXPECTED_CLASSES:
        mark = "✅" if class_totals.get(cls, 0) > 0 else "❌"
        print(f"  {mark}  {cls:<22} {class_totals.get(cls, 0):>6} labels")

    # Unknown classes
    all_seen = {cname for v in all_data.values()
                for s in v for (_, cname, *_) in s["labels"]}
    unknown = all_seen - set(EXPECTED_CLASSES)
    if unknown:
        print(f"\n  ⚠  Classes in dataset not in schema: {unknown}")

    # Balance
    present = {c: n for c, n in class_totals.items() if n > 0}
    if len(present) >= 2:
        max_c = max(present, key=present.get)
        min_c = min(present, key=present.get)
        ratio = present[max_c] / present[min_c]
        print(f"\n── Balance ──────────────────────────────────────────────")
        print(f"  Most  : {max_c} ({present[max_c]})")
        print(f"  Least : {min_c} ({present[min_c]})")
        flag = "⚠  Large imbalance" if ratio > 5 else "✅  Acceptable"
        print(f"  Ratio : {ratio:.1f}×  {flag}")

    print("=" * 62 + "\n")
    return missing, class_totals


# ── Visuals ───────────────────────────────────────────────────────────────────

def show_grid(all_data: dict, n: int, filter_cls: str | None):
    all_samples = [s for v in all_data.values() for s in v]

    if filter_cls:
        all_samples = [s for s in all_samples
                       if any(cname == filter_cls
                              for (_, cname, *_) in s["labels"])]
        title = f"Labelled images — class: {filter_cls}"
    else:
        title = "Labelled images — random sample"

    if not all_samples:
        print(f"  No images found for class: {filter_cls}")
        return

    chosen = random.sample(all_samples, min(n, len(all_samples)))
    ncols  = min(4, len(chosen))
    nrows  = (len(chosen) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    axes = np.array(axes).flatten()

    for ax, s in zip(axes, chosen):
        img = cv2.imread(str(s["img_path"]))
        if img is None:
            ax.axis("off")
            continue
        ann = draw_boxes(img, s["labels"])
        ax.imshow(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
        cls_list = sorted({cname for (_, cname, *_) in s["labels"]})
        ax.set_title(", ".join(cls_list), fontsize=8)
        ax.axis("off")

    for ax in axes[len(chosen):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_crop_mosaic(all_data: dict, target_classes: list, n_per_class: int):
    """Zoomed-in crop per bounding box — good for spotting mislabels."""
    all_samples = [s for v in all_data.values() for s in v]
    random.shuffle(all_samples)

    for cls_name in target_classes:
        crops = []
        for s in all_samples:
            if len(crops) >= n_per_class:
                break
            img = cv2.imread(str(s["img_path"]))
            if img is None:
                continue
            ih, iw = img.shape[:2]
            for (_, cname, xc, yc, w, h) in s["labels"]:
                if cname != cls_name:
                    continue
                x1 = max(0, int((xc - w / 2) * iw) - 10)
                y1 = max(0, int((yc - h / 2) * ih) - 10)
                x2 = min(iw, int((xc + w / 2) * iw) + 10)
                y2 = min(ih, int((yc + h / 2) * ih) + 10)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop = cv2.resize(crop, (128, 128))
                crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                if len(crops) >= n_per_class:
                    break

        if not crops:
            print(f"  No crops found for: {cls_name}")
            continue

        ncols = min(8, len(crops))
        nrows = (len(crops) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(2.2 * ncols, 2.2 * nrows))
        fig.suptitle(f"Crop mosaic — {cls_name}  ({len(crops)} samples)",
                     fontsize=12, fontweight="bold")
        axes = np.array(axes).flatten()
        for ax, crop in zip(axes, crops):
            ax.imshow(crop)
            ax.axis("off")
        for ax in axes[len(crops):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()


# ── Entry ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   required=True,
                        help="Roboflow YOLO export folder or .zip file")
    parser.add_argument("--split",     default=None,
                        choices=["train", "val", "test"],
                        help="Only check one split (default: all)")
    parser.add_argument("--class",     dest="filter_cls", default=None,
                        help="Filter grid to one class name")
    parser.add_argument("--grid",      default=12, type=int,
                        help="Images in grid view (default 12)")
    parser.add_argument("--mosaic",    default=16, type=int,
                        help="Crop patches per class in mosaic (default 16)")
    parser.add_argument("--no-show",   action="store_true",
                        help="Print report only, skip visual windows")
    args = parser.parse_args()

    root = unzip_if_needed(Path(args.dataset))
    print(f"\nDataset root : {root}")

    names, split_dirs = load_yaml(root)
    print(f"Classes ({len(names)}) : {names}")

    # Schema check
    if sorted(names) != sorted(EXPECTED_CLASSES):
        print(f"\n⚠  Class mismatch with expected schema!")
        print(f"   Expected : {sorted(EXPECTED_CLASSES)}")
        print(f"   Got      : {sorted(names)}")
    else:
        print("✅  Classes match expected schema.")

    # Load splits
    target_splits = [args.split] if args.split else ["train", "val", "test"]
    all_data      = {}

    for split in target_splits:
        img_dir = split_dirs.get(split)
        if img_dir is None:
            print(f"\n  {split}: not in data.yaml, skipping.")
            all_data[split] = []
            continue
        print(f"\nLoading {split} …")
        samples = load_split(img_dir, names)
        print(f"  → {len(samples)} images with valid labels")
        all_data[split] = samples

    # Report
    missing, class_totals = print_report(all_data, names)

    if args.no_show:
        return

    # Grid view
    print("Opening image grid …")
    show_grid(all_data, args.grid, args.filter_cls)

    # Crop mosaics — always show left_turn + speed_sign, plus any missing classes
    mosaic_targets = ["left_turn", "speed_sign"]
    for cls in EXPECTED_CLASSES:
        if class_totals.get(cls, 0) > 0 and cls not in mosaic_targets:
            mosaic_targets.append(cls)

    print("Opening crop mosaics …")
    show_crop_mosaic(all_data, mosaic_targets, args.mosaic)


if __name__ == "__main__":
    main()
