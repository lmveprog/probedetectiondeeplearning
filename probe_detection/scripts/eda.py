"""Exploratory data analysis of the probe dataset.

Produces the figures used in the report:
  - bbox center heatmap + size/aspect-ratio distributions
  - per-flight image counts (drives the grouped train/val split)
  - a contact sheet of annotated samples

Usage: python scripts/eda.py [--dataset ../probe_dataset] [--out reports/eda]
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def flight_key(file_name: str) -> str:
    """Group key for one physical flight/sequence.

    File names look like  E300SA22440034_00447_133_1flight_2400_2.jpg:
    <drone-serial>_<flight-counter>_<seq>_1flight_<timestamp-ms>_<cam>.
    Frames sharing <serial>_<flight-counter> come from the same flight and
    are near-duplicates; they must never be split across train and val.
    """
    parts = file_name.split("_")
    return f"{parts[0]}_{parts[1]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).resolve().parents[2] / "probe_dataset"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "reports" / "eda"))
    args = ap.parse_args()

    dataset = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads((dataset / "probe_labels.json").read_text())
    images = {im["id"]: im for im in data["images"]}
    anns = data["annotations"]

    # ---- basic stats -----------------------------------------------------
    sizes = Counter((im["width"], im["height"]) for im in images.values())
    flights = Counter(flight_key(im["file_name"]) for im in images.values())
    serials = Counter(im["file_name"].split("_")[0] for im in images.values())
    print(f"{len(images)} images, {len(anns)} annotations")
    print(f"image sizes: {dict(sizes)}")
    print(f"{len(flights)} flights across {len(serials)} drones")
    for s, n in serials.most_common():
        print(f"  drone {s}: {n} images")

    # ---- bbox geometry ---------------------------------------------------
    cx, cy, ws, hs, areas = [], [], [], [], []
    for a in anns:
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        cx.append((x + w / 2) / im["width"])
        cy.append((y + h / 2) / im["height"])
        ws.append(w / im["width"])
        hs.append(h / im["height"])
        areas.append(w * h / (im["width"] * im["height"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].hist2d(cx, cy, bins=24, range=[[0, 1], [0, 1]], cmap="viridis")
    axes[0].invert_yaxis()
    axes[0].set(title="Probe center heatmap", xlabel="x (norm.)", ylabel="y (norm.)")
    axes[1].scatter(ws, hs, s=12, alpha=0.5)
    axes[1].set(title="Bbox size (normalized)", xlabel="width", ylabel="height")
    axes[2].hist(areas, bins=30)
    axes[2].set(title="Bbox area fraction of image", xlabel="area fraction", ylabel="count")
    fig.tight_layout()
    fig.savefig(out / "bbox_stats.png", dpi=130)
    print(f"probe area: min {min(areas):.3f}, median {np.median(areas):.3f}, max {max(areas):.3f}")

    # ---- per-flight counts (motivates the grouped split) ------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    labels, counts = zip(*flights.most_common())
    ax.bar(range(len(counts)), counts)
    ax.set(title=f"Images per flight ({len(flights)} flights)", xlabel="flight", ylabel="images")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "images_per_flight.png", dpi=130)

    # ---- contact sheet of annotated samples ------------------------------
    rng = random.Random(0)
    by_flight = defaultdict(list)
    for a in anns:
        by_flight[flight_key(images[a["image_id"]]["file_name"])].append(a)
    # one sample per flight for maximum visual diversity
    picks = [rng.choice(v) for v in by_flight.values()]
    rng.shuffle(picks)
    picks = picks[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    for ax, a in zip(axes.flat, picks):
        im = images[a["image_id"]]
        img = cv2.cvtColor(cv2.imread(str(dataset / "probe_images" / im["file_name"])), cv2.COLOR_BGR2RGB)
        x, y, w, h = (int(v) for v in a["bbox"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 60, 60), 3)
        ax.imshow(img)
        ax.set_title(flight_key(im["file_name"]), fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "annotated_samples.png", dpi=130)
    print(f"figures saved to {out}")


if __name__ == "__main__":
    main()
