import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List

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
    images = list_flat_images(args.root)
    assert len(images) == 700, f"len(images):{len(images)} != 700"


    folders = [
        "affection",
        "composition",
        "entity",
        "imagination",
        "long_text",
        "style",
        "text_rendering",
    ]
    
    for folder in folders:
        os.makedirs(os.path.join(args.root, folder))

    for img in images:
        num = int(img[:-4])
        folder = folders[num // 100]
        idx = num % 100
        shutil.move(os.path.join(args.root, img), os.path.join(args.root, folder, f"{idx}.png"))