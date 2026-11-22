#!/usr/bin/env python3
"""Prepare NEU-DET (KeenForgeAI/NEU-DET-corrected) for YOLO training.

Reads Pascal VOC XML annotations + the dataset's included ImageSets split
(train 1437 / val 180 / test 180 = 8:1:1 of 1797 images) and writes
YOLO-format labels + image copies under data/yolo/, plus data/neu_det.yaml.

Class names are discovered from the XML files themselves, so the label
order is deterministic (alphabetical).
"""
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        b = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(b.find(t).text) for t in ("xmin", "ymin", "xmax", "ymax"))
        # YOLO format: class x_center y_center width height (normalized)
        x_c = ((xmin + xmax) / 2) / w
        y_c = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        boxes.append((CLASSES.index(name), x_c, y_c, bw, bh))
    return boxes


def main():
    # sanity: discover class names actually present in the XMLs
    found = set()
    for xml in (RAW / "annotations").glob("*.xml"):
        for obj in ET.parse(xml).getroot().findall("object"):
            found.add(obj.find("name").text.strip())
    assert found == set(CLASSES), f"unexpected class names: {found ^ set(CLASSES)}"
    print(f"classes confirmed: {CLASSES}")

    counts = {}
    for split in ("train", "val", "test"):
        names = (RAW / "ImageSets" / f"{split}.txt").read_text().split()
        img_dir = OUT / split / "images"
        lbl_dir = OUT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for stem in names:
            src_img = RAW / "images" / split / f"{stem}.jpg"
            if not src_img.exists():
                # fallback: flat layout
                src_img = RAW / "images" / f"{stem}.jpg"
            assert src_img.exists(), f"missing image {src_img}"
            shutil.copy2(src_img, img_dir / f"{stem}.jpg")
            boxes = parse_xml(RAW / "annotations" / f"{stem}.xml")
            with open(lbl_dir / f"{stem}.txt", "w") as f:
                for cls, xc, yc, bw, bh in boxes:
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        counts[split] = len(names)
        print(f"{split}: {len(names)} images")

    yaml = ROOT / "data" / "neu_det.yaml"
    yaml.write_text(
        f"# NEU-DET (KeenForgeAI/NEU-DET-corrected), 1797 images, 200x200 grayscale\n"
        f"path: {ROOT / 'data' / 'yolo'}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n\n"
        f"names:\n"
        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES))
    )
    print(f"wrote {yaml}")
    print(f"total: {sum(counts.values())} images -> {OUT}")


if __name__ == "__main__":
    main()
