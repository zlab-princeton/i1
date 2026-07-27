import argparse
import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image
from tqdm import tqdm


def load_metadata() -> List[List[str]]:
    metadata_path = Path(__file__).resolve().parent / "dpg_bench.json"
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_images(image_paths: Iterable[Path]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in image_paths:
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def stitch_2x2(images: List[Image.Image]) -> Image.Image:
    if len(images) != 4:
        raise ValueError(f"Expected 4 images, got {len(images)}")
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 2, height * 2))
    positions = [(0, 0), (width, 0), (0, height), (width, height)]
    for img, (x, y) in zip(images, positions):
        canvas.paste(img, (x, y))
    return canvas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser()
    if not root.is_dir():
        raise NotADirectoryError(f"{root} is not a directory")

    metadata = load_metadata()
    processed_dir = root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for idx, entry in tqdm(
        enumerate(metadata), total=len(metadata), desc="Processing folders"
    ):
        out_name = entry[0]
        sample_dir = root / f"{idx:05d}" / "samples"
        image_paths = [sample_dir / f"{i:05d}.png" for i in range(4)]

        if not all(p.is_file() for p in image_paths):
            missing = [str(p) for p in image_paths if not p.is_file()]
            raise FileNotFoundError(
                f"Missing expected images for index {idx}: {', '.join(missing)}"
            )

        images = load_images(image_paths)
        combined = stitch_2x2(images)
        out_path = processed_dir / f"{out_name}.png"
        combined.save(out_path)