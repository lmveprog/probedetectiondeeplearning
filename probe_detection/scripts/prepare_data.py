"""Convert the COCO-style probe dataset to YOLO format with a leakage-free split.

Frames from the same flight are near-duplicates, so the train/val split is
done at the *flight* level (see eda.py:flight_key), not at the image level.
A random image-level split would leak near-identical frames into validation
and report misleadingly high metrics.

The val set is built greedily: flights are shuffled per drone and picked
round-robin across drones until ~20% of images are in val, so every drone
contributes validation flights when possible.

Output layout (YOLO):
  data/images/{train,val}/*.jpg   (symlinks to the original images)
  data/labels/{train,val}/*.txt   (class cx cy w h, normalized)
  data/probe.yaml

Usage: python scripts/prepare_data.py [--dataset ../probe_dataset] [--val-frac 0.2]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from eda import flight_key

VAL_SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "probe_dataset"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)

    data = json.loads((dataset / "probe_labels.json").read_text())
    images = {im["id"]: im for im in data["images"]}
    anns_by_image = defaultdict(list)
    for a in data["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    # ---- group flights by drone, pick val flights round-robin -------------
    flights_by_drone = defaultdict(set)
    images_by_flight = defaultdict(list)
    for im in images.values():
        fk = flight_key(im["file_name"])
        flights_by_drone[im["file_name"].split("_")[0]].add(fk)
        images_by_flight[fk].append(im)

    rng = random.Random(VAL_SEED)
    queues = {d: rng.sample(sorted(f), len(f)) for d, f in flights_by_drone.items()}
    target = args.val_frac * len(images)
    val_flights, val_count = set(), 0
    while val_count < target:
        for drone in sorted(queues):
            if val_count >= target or not queues[drone]:
                continue
            fk = queues[drone].pop()
            val_flights.add(fk)
            val_count += len(images_by_flight[fk])

    # ---- write YOLO structure ---------------------------------------------
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for im in images.values():
        split = "val" if flight_key(im["file_name"]) in val_flights else "train"
        counts[split] += 1
        link = out / "images" / split / im["file_name"]
        if not link.exists():
            link.symlink_to((dataset / "probe_images" / im["file_name"]).resolve())
        lines = []
        for a in anns_by_image[im["id"]]:
            x, y, w, h = a["bbox"]
            cx, cy = (x + w / 2) / im["width"], (y + h / 2) / im["height"]
            lines.append(f"0 {cx:.6f} {cy:.6f} {w / im['width']:.6f} {h / im['height']:.6f}")
        (out / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines) + "\n")

    (out / "probe.yaml").write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n  0: probe\n"
    )

    print(f"train: {counts['train']} images | val: {counts['val']} images")
    print(f"val flights ({len(val_flights)}): {sorted(val_flights)}")


if __name__ == "__main__":
    main()
