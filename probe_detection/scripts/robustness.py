"""Robustness study: recall under realistic image degradations.

Validation images are corrupted with four degradations an inspection drone
actually encounters — darkness, motion blur, sensor noise, overexposure —
at four severity levels each, and recall (IoU>=0.5 at the calibrated 0.30
threshold) is measured per (corruption, severity). The resulting curves say
*when* the detector can be trusted, not just how good it is on average.

Usage: python scripts/robustness.py [--weights weights/probe_yolo11n.pt] [--device cpu]
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]

SEVERITIES = [1, 2, 3, 4]


def darken(img, s):
    return (img.astype(np.float32) * (1 - 0.18 * s)).clip(0, 255).astype(np.uint8)


def overexpose(img, s):
    return (img.astype(np.float32) * (1 + 0.45 * s)).clip(0, 255).astype(np.uint8)


def motion_blur(img, s):
    k = 2 * s + 1
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1 / k  # horizontal streak, like lateral drone motion
    return cv2.filter2D(img, -1, kernel)


def sensor_noise(img, s):
    noise = np.random.default_rng(0).normal(0, 6 * s, img.shape)
    return (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)


CORRUPTIONS = {"darkness": darken, "overexposure": overexpose,
               "motion blur": motion_blur, "sensor noise": sensor_noise}


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights" / "probe_yolo11n.pt"))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval_yolo11n"))
    args = ap.parse_args()

    model = YOLO(args.weights)

    items = []
    for img_path in sorted((ROOT / "data" / "images" / "val").iterdir()):
        label = ROOT / "data" / "labels" / "val" / (img_path.stem + ".txt")
        cx, cy, w, h = [float(v) for v in label.read_text().split()[1:5]]
        img = cv2.imread(str(img_path))
        H, W = img.shape[:2]
        gt = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
        items.append((img, gt))

    def recall(images_gts):
        hits = 0
        for img, gt in images_gts:
            res = model.predict(img, conf=args.conf, verbose=False, device=args.device)[0]
            if len(res.boxes):
                best = int(res.boxes.conf.argmax())
                if iou_xyxy([float(v) for v in res.boxes.xyxy[best]], gt) >= 0.5:
                    hits += 1
        return hits / len(images_gts)

    baseline = recall(items)
    curves = {name: [recall([(fn(im, s), gt) for im, gt in items]) for s in SEVERITIES]
              for name, fn in CORRUPTIONS.items()}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, ys in curves.items():
        ax.plot([0] + SEVERITIES, [baseline] + ys, "o-", label=name)
    ax.set(xlabel="corruption severity", ylabel=f"recall (IoU≥0.5, conf≥{args.conf})",
           title="Recall under realistic degradations", ylim=(0, 1))
    ax.set_xticks([0] + SEVERITIES)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.out)
    fig.savefig(out / "robustness.png", dpi=130)

    summary = {"baseline_recall": round(baseline, 4),
               **{k: [round(v, 4) for v in ys] for k, ys in curves.items()}}
    (out / "robustness.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
