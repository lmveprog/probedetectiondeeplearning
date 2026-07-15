"""Generate probe-free (negative) training images to reduce false alarms.

The dataset contains no negative examples, yet the system must report
"no probe". Two kinds of synthetic negatives are generated from TRAIN
images only (validation flights stay untouched):

  1. Inpainted full frames: the probe is erased with OpenCV Telea
     inpainting (mask = dilated GT box). In the mostly dark scenes this
     yields visually plausible probe-free frames with full navigation
     context. One frame per training flight to avoid near-duplicates.
  2. Probe-free crops: the largest strip outside the GT box (same logic
     as evaluate.py), which is artifact-free by construction and guards
     against the model learning inpainting artifacts as a shortcut.

Negatives are written to data/images/train_neg with empty label files,
and data/probe_neg.yaml lists [train, train_neg] as training sources
(~13% background images, in line with common YOLO practice).

Usage: python scripts/make_negatives.py
"""

import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from eda import flight_key

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MIN_NEG_SIDE = 160
CROPS_PER_FLIGHT = 1


def read_gt_xyxy(img_path: Path):
    label = DATA / "labels" / "train" / (img_path.stem + ".txt")
    cx, cy, w, h = [float(v) for v in label.read_text().split()[1:5]]
    im = cv2.imread(str(img_path))
    H, W = im.shape[:2]
    return im, [int((cx - w / 2) * W), int((cy - h / 2) * H), int((cx + w / 2) * W), int((cy + h / 2) * H)]


def main() -> None:
    out_img = DATA / "images" / "train_neg"
    out_lbl = DATA / "labels" / "train_neg"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    by_flight = defaultdict(list)
    for p in sorted((DATA / "images" / "train").iterdir()):
        by_flight[flight_key(p.name)].append(p)

    rng = random.Random(0)
    n_inpaint = n_crop = 0
    for flight, paths in sorted(by_flight.items()):
        picks = rng.sample(paths, min(2, len(paths)))

        # 1. inpainted full frame
        img, (x1, y1, x2, y2) = read_gt_xyxy(picks[0])
        mask = np.zeros(img.shape[:2], np.uint8)
        pad = 8  # dilate the mask so the probe's bright halo is erased too
        mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 255
        inpainted = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        # Telea output is unnaturally smooth; re-inject sensor noise matched to
        # the rest of the frame so the patch is not a learnable artifact.
        residual = img.astype(np.float32) - cv2.medianBlur(img, 5).astype(np.float32)
        noise_std = float(residual[mask == 0].std())
        noise = np.random.default_rng(0).normal(0, noise_std, img.shape).astype(np.float32)
        blended = inpainted.astype(np.float32) + noise
        inpainted = np.where(mask[..., None] > 0, blended.clip(0, 255), inpainted).astype(np.uint8)
        name = f"neg_inpaint_{picks[0].stem}.jpg"
        cv2.imwrite(str(out_img / name), inpainted)
        (out_lbl / (Path(name).stem + ".txt")).write_text("")
        n_inpaint += 1

        # 2. probe-free crop (largest strip outside the box)
        img, (x1, y1, x2, y2) = read_gt_xyxy(picks[-1])
        H, W = img.shape[:2]
        strips = [(0, 0, x1, H), (x2, 0, W, H), (0, 0, W, y1), (0, y2, W, H)]
        sx1, sy1, sx2, sy2 = max(strips, key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
        if sx2 - sx1 >= MIN_NEG_SIDE and sy2 - sy1 >= MIN_NEG_SIDE:
            name = f"neg_crop_{picks[-1].stem}.jpg"
            cv2.imwrite(str(out_img / name), img[sy1:sy2, sx1:sx2])
            (out_lbl / (Path(name).stem + ".txt")).write_text("")
            n_crop += 1

    (DATA / "probe_neg.yaml").write_text(
        f"path: {DATA.resolve()}\n"
        "train:\n  - images/train\n  - images/train_neg\n"
        "val: images/val\n"
        "names:\n  0: probe\n"
    )
    n_train = len(list((DATA / "images" / "train").iterdir()))
    print(f"{n_inpaint} inpainted + {n_crop} cropped negatives "
          f"({n_inpaint + n_crop} total, {100 * (n_inpaint + n_crop) / n_train:.0f}% of train)")


if __name__ == "__main__":
    main()
