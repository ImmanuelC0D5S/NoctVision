"""
Dataset Preparation Script for NoctVision.
Converts raw LLVIP dataset (visible camera images & XML annotations) into
standard YOLO format with offline Albumentations augmentation.

LLVIP raw layout expected:
    data/raw/
        visible/
            train/        # .jpg images
            test/         # .jpg images (optional, merged if present)
        Annotations/      # Pascal VOC .xml files
            010001.xml
            ...

Output layout:
    data/processed/
        images/
            train/        # original + augmented images
            val/
            test/
        labels/
            train/        # original + augmented YOLO .txt labels
            val/
            test/
"""

import os
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from augmentations import get_offline_augmentations, get_noctvision_transforms


# ---------------------------------------------------------------------------
# 1. XML → YOLO conversion
# ---------------------------------------------------------------------------

def parse_xml_annotation(xml_path: str) -> list[tuple[int, float, float, float, float]]:
    """
    Parse a single Pascal VOC XML annotation file.

    Returns list of (class_id, x_center, y_center, width, height) in YOLO
    normalized format.  LLVIP images are 1280×1024.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        if name != "person":
            continue

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        xmax = float(bndbox.find("xmax").text)
        ymin = float(bndbox.find("ymin").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = ((xmin + xmax) / 2.0) / img_w
        y_center = ((ymin + ymax) / 2.0) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        # Clamp to (0, 1] — Albumentations YOLO format excludes 0
        x_center = max(1e-6, min(1.0, x_center))
        y_center = max(1e-6, min(1.0, y_center))
        width = max(1e-6, min(1.0, width))
        height = max(1e-6, min(1.0, height))

        # Skip degenerate boxes
        if width < 1e-5 or height < 1e-5:
            continue

        boxes.append((0, x_center, y_center, width, height))  # class 0 = person

    return boxes


def write_yolo_label(label_path: str, boxes: list[tuple[int, float, float, float, float]]):
    """Write YOLO-format label file."""
    with open(label_path, "w") as f:
        for cls, xc, yc, w, h in boxes:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def yolo_boxes_from_albumentations(bboxes: list[list], class_labels: list[int]) -> list[tuple[int, float, float, float, float]]:
    """
    Reconstruct YOLO tuple list after Albumentations bbox transforms.
    Albumentations returns bboxes as [[x_center, y_center, w, h], ...].
    """
    result = []
    for bbox, cls in zip(bboxes, class_labels):
        xc, yc, w, h = bbox
        result.append((cls, xc, yc, w, h))
    return result


# ---------------------------------------------------------------------------
# 2. Dataset discovery — find all image↔annotation pairs
# ---------------------------------------------------------------------------

def discover_pairs(
    raw_dir: Path,
) -> dict[str, dict]:
    """
    Walk the raw LLVIP directory and return a dict mapping
    image_stem → { "image": Path, "xml": Path|None, "source_split": str }.

    Source split comes from the folder the image was found in (train / test).
    """
    pairs: dict[str, dict] = {}

    # Possible image roots (LLVIP can ship as visible/train or just train/)
    image_roots = []
    for candidate in [
        raw_dir / "visible" / "train",
        raw_dir / "visible" / "test",
        raw_dir / "train",
        raw_dir / "test",
    ]:
        if candidate.is_dir():
            image_roots.append(candidate)

    # Annotation root
    anno_dir: Optional[Path] = None
    for candidate in [
        raw_dir / "Annotations",
        raw_dir / "annotations",
        raw_dir / "labels",
    ]:
        if candidate.is_dir():
            anno_dir = candidate
            break

    if not image_roots:
        raise FileNotFoundError(
            f"No image directories found under {raw_dir}. "
            "Expected data/raw/visible/train/ or data/raw/train/."
        )

    for img_root in image_roots:
        source_split = img_root.name  # "train" or "test"
        for img_file in sorted(img_root.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            stem = img_file.stem
            xml_path = anno_dir / f"{stem}.xml" if anno_dir else None
            if xml_path and not xml_path.exists():
                xml_path = None

            # Keep first occurrence (prefer train over test if duplicated)
            if stem not in pairs:
                pairs[stem] = {
                    "image": img_file,
                    "xml": xml_path,
                    "source_split": source_split,
                }

    return pairs


# ---------------------------------------------------------------------------
# 3. Train / Val / Test split
# ---------------------------------------------------------------------------

def split_dataset(
    pairs: dict[str, dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Randomly assign each image stem to train / val / test.
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    stems = list(pairs.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(stems)

    n = len(stems)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }


# ---------------------------------------------------------------------------
# 4. Core pipeline — copy originals + apply offline augmentations
# ---------------------------------------------------------------------------

def build_processed_dataset(
    raw_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_augmentations: int = 2,
    img_size: int = 640,
    seed: int = 42,
):
    """
    Main entry point.

    1. Discover image↔annotation pairs in raw_dir.
    2. Split into train / val / test.
    3. Copy original images + labels to output_dir.
    4. Apply offline augmentations to training set only.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)

    # Create output directories
    for split in ("train", "val", "test"):
        (out_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # --- Discover pairs ---
    print("Discovering image–annotation pairs ...")
    pairs = discover_pairs(raw_path)
    print(f"  Found {len(pairs)} pairs.")

    # --- Split ---
    splits = split_dataset(pairs, train_ratio, val_ratio, seed)
    for split_name, stems in splits.items():
        print(f"  {split_name}: {len(stems)} images")

    # --- Augmentation pipelines (training only) ---
    aug_pipelines = get_offline_augmentations(img_size=img_size, num_augmentations=num_augmentations)
    val_pipeline = get_noctvision_transforms(img_size=img_size, is_train=False)

    # --- Process each split ---
    for split_name, stems in splits.items():
        is_train = split_name == "train"
        desc = f"Processing {split_name}"

        for stem in tqdm(stems, desc=desc):
            pair = pairs[stem]
            img_path = pair["image"]
            xml_path = pair["xml"]

            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  Warning: could not read {img_path}, skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Parse annotations (or empty if no XML)
            if xml_path and xml_path.exists():
                yolo_boxes = parse_xml_annotation(str(xml_path))
            else:
                yolo_boxes = []

            # -----------------------------------------------------------
            # A) Copy ORIGINAL (resized to img_size, label unchanged)
            # -----------------------------------------------------------
            if yolo_boxes:
                bboxes = [list(b[1:]) for b in yolo_boxes]
                class_labels = [b[0] for b in yolo_boxes]
            else:
                bboxes = []
                class_labels = []

            if is_train:
                augmented = aug_pipelines[0](
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
            else:
                augmented = val_pipeline(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )

            aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            aug_boxes = yolo_boxes_from_albumentations(
                augmented["bboxes"], augmented["class_labels"]
            )

            out_img = out_path / "images" / split_name / f"{stem}.jpg"
            out_lbl = out_path / "labels" / split_name / f"{stem}.txt"
            cv2.imwrite(str(out_img), aug_img)
            write_yolo_label(str(out_lbl), aug_boxes)

            # -----------------------------------------------------------
            # B) Generate AUGMENTED copies (training only)
            # -----------------------------------------------------------
            if is_train:
                for aug_idx, pipeline in enumerate(aug_pipelines[1:], start=2):
                    augmented = pipeline(
                        image=image,
                        bboxes=bboxes,
                        class_labels=class_labels,
                    )
                    aug_img = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
                    aug_boxes = yolo_boxes_from_albumentations(
                        augmented["bboxes"], augmented["class_labels"]
                    )

                    aug_stem = f"{stem}_aug{aug_idx}"
                    cv2.imwrite(
                        str(out_path / "images" / split_name / f"{aug_stem}.jpg"),
                        aug_img,
                    )
                    write_yolo_label(
                        str(out_path / "labels" / split_name / f"{aug_stem}.txt"),
                        aug_boxes,
                    )

    # --- Summary ---
    total_images = sum(
        len(list((out_path / "images" / s).glob("*")))
        for s in ("train", "val", "test")
    )
    print(f"\nDone. {total_images} total images written to {out_path}")
    print("  Structure:")
    for split in ("train", "val", "test"):
        n = len(list((out_path / "images" / split).glob("*")))
        print(f"    images/{split}: {n}  |  labels/{split}: {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare LLVIP Visible Dataset for YOLOv8 (with offline augmentation)"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Path to raw LLVIP dataset (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Path to output processed dataset (default: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)",
    )
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=2,
        help="Number of augmented copies per training image (default: 2)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Output image size (default: 640)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split (default: 42)",
    )
    args = parser.parse_args()

    build_processed_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        num_augmentations=args.num_augmentations,
        img_size=args.img_size,
        seed=args.seed,
    )
