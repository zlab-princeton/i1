import argparse
import os
import json
import re
import shutil
from pathlib import Path
from typing import List
METADATA_DIR = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../jax/inference/prompts/CVTG-2K.json"))

def list_flat_images(root: Path) -> List[Path]:
    imgs = []
    for p in os.listdir(root):
        if not os.path.isfile(os.path.join(root, p)):
            continue
        if not p.endswith(".png"):
            continue
        # accept numeric stems like 00000
        if re.fullmatch(r"\d+", p[:-4]):
            imgs.append(p)

    imgs.sort(key=lambda x: int(x[:-4]))
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.root)
    with open(METADATA_DIR, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    images = list_flat_images(args.root)
    assert len(images) == len(mapping), f"len(images):{len(images)} != len(mapping):{len(mapping)}"

    for cat in ["CVTG", "CVTG-Style"]:
        for k in [2, 3, 4, 5]:
            os.makedirs(os.path.join(args.root, cat, str(k)), exist_ok=True)

    for src_img, (key, prompt) in zip(images, mapping):
        category, k, idx = key.split("_")
        shutil.move(os.path.join(args.root, src_img), os.path.join(args.root, category, str(k), f"{idx}.png"))