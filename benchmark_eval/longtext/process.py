import argparse
import os
import json
import shutil
from pathlib import Path
from typing import List

def list_images(root: Path) -> List[Path]:
    imgs = []
    for p in os.listdir(root):
        if not os.path.isfile(os.path.join(root, p)):
            continue
        if not p.endswith(".png"):
            continue
        imgs.append(p)
    return imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    assert os.path.exists(args.root)
    row_to_prompt_id = {}
    with open("text_prompts.jsonl", "r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            obj = json.loads(line)
            row_to_prompt_id[row_idx] = obj["prompt_id"]
    images = list_images(args.root)
    assert len(images) == 640, f"len(images):{len(images)} != 640"
    assert len(row_to_prompt_id) == 160, f"len(mapping):{len(row_to_prompt_id)} != 160"

    for src_img in images:
        img_id, repeat_count_and_suffix = src_img.split("_")
        shutil.move(os.path.join(args.root, src_img), os.path.join(args.root, f"{row_to_prompt_id[int(img_id)]}_{repeat_count_and_suffix}"))