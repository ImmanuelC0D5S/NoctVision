"""
Dataset Preparation Script for NoctVision.
Converts raw LLVIP dataset (visible camera images & annotations) into standard YOLO structure.
"""

import os
import shutil
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def voc_to_yolo(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_w = int(root.find("size/width").text)
    img_h = int(root.find("size/height").text)

    lines = []
    for obj in root.iter("object"):
        xmin = float(obj.find("bndbox/xmin").text)
        ymin = float(obj.find("bndbox/ymin").text)
        xmax = float(obj.find("bndbox/xmax").text)
        ymax = float(obj.find("bndbox/ymax").text)

        x_center = (xmin+xmax)/(2*img_w)
        y_center = (ymin+ymax)/(2*img_h)
        w = (xmax-xmin)/img_w
        h = (ymax-ymin)/img_h

        lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}") # yolo format contains all the metrics
        # in a normalised format between 0 and 1, hence dividing by the width
    return lines


def prepare_llvip_visible(raw_dir: str, output_dir: str, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """
    Parses raw LLVIP visible dataset and prepares YOLO format layout.
    
    Structure created in output_dir:
        images/train, images/val, images/test
        labels/train, labels/val, labels/test
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    
    visible_img_dir = raw_path / "visible" / "train"  # LLVIP visible channel folder
    labels_dir = raw_path / "labels"
    
    print(f"Processing visible camera images from: {raw_path}")
    print(f"Output directory: {output_path}")

    # Create directory structure
    for split in ["train", "val", "test"]:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print("Dataset directory structure initialized.")

def convert_split(raw_dir: str, annotation_dir: str, output_dir: str, split):
    image_folder = Path(raw_dir) / "visible" / split # the source file is from the raw directory + the split 
    for img_path in sorted(image_folder.glob("*.jpg")):
        xml_path= Path(annotation_dir) / (img_path.stem + ".xml") # the corresponding xml file which has all the metrics
        lines=voc_to_yolo(xml_path) #calculcate using the above function
        label_path=Path(output_dir) / "labels" / split / (img_path.stem + ".txt") #store it in the LABELS directory 
        with open(label_path,"w") as f:
            f.write(("\n").join(lines))
        shutil.copy(img_path, Path(output_dir) / "images" /split / img_path.name)

def val_split_maker(data_dir:str , videoids:list):
    datapath= Path(data_dir)
    movedvideos=0
    for image_path in sorted((datapath/"images"/"train").glob("*.jpg")):
        if image_path.stem[:2] in videoids: # the stem returns the full video id of which the first two is the non frame vid id
            label_path = (datapath/"labels"/"train"/(image_path.stem + ".txt"))
            shutil.move(image_path,(datapath/"images"/"val"/image_path.name))
            shutil.move(label_path,(datapath/"labels"/"val"/(image_path.stem +".txt")))
            movedvideos+=1
    return f"{movedvideos} frames and text files moved to val folder"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LLVIP Visible Dataset for YOLOv8")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw LLVIP dataset")
    parser.add_argument("--ann-dir", type=str, default="LLVIP/Annotations", help="Path to LLVIP annotation XMLs")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to output processed dataset")
    args = parser.parse_args()

    prepare_llvip_visible(args.raw_dir, args.output_dir)
    convert_split(args.raw_dir, args.ann_dir, args.output_dir, "train")
    convert_split(args.raw_dir, args.ann_dir, args.output_dir, "test")
    print(val_split_maker(args.output_dir, ["12", "03", "04"]))

