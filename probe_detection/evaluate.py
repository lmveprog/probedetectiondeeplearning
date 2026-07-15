"""Evaluation of the trained probe detector.

Four parts, each feeding a section of the report:
  1. Standard detection metrics on the val split (precision, recall, F1,
     mAP@0.5, mAP@0.5:0.95) via ultralytics, plus mean IoU of matched boxes.
  2. "No probe" calibration: the dataset has no negative images, so we build
     synthetic negatives by cropping, from each val image, the largest strip
     that does NOT contain the probe. These realistic probe-free industrial
     backgrounds let us measure the false-alarm rate and sweep the confidence
     threshold to pick the operating point used by inference.py.
  3. Inference runtime benchmark (per-image latency on MPS and CPU).
  4. Qualitative grid of the best and worst val predictions (by IoU).

Usage: python evaluate.py [--weights runs/yolo11n/weights/best.pt]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MIN_NEG_SIDE = 160  # a synthetic negative strip must be at least this wide/tall


def iou_xyxy(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def load_val_set():
    """Return [(image_path, gt_xyxy)] for the val split."""
    items = []
    for img_path in sorted((ROOT / "data" / "images" / "val").iterdir()):
        label = ROOT / "data" / "labels" / "val" / (img_path.stem + ".txt")
        cx, cy, w, h = [float(v) for v in label.read_text().split()[1:5]]
        im = cv2.imread(str(img_path))
        H, W = im.shape[:2]
        gt = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
        items.append((img_path, gt))
    return items


def probe_free_crop(img, gt):
    """Largest strip of img strictly outside the probe bbox, or None if too small."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in gt)
    strips = {  # (x1, y1, x2, y2) of the four regions around the bbox
        "left": (0, 0, x1, H), "right": (x2, 0, W, H),
        "top": (0, 0, W, y1), "bottom": (0, y2, W, H),
    }
    best = max(strips.values(), key=lambda s: (s[2] - s[0]) * (s[3] - s[1]))
    if best[2] - best[0] < MIN_NEG_SIDE or best[3] - best[1] < MIN_NEG_SIDE:
        return None
    return img[best[1]:best[3], best[0]:best[2]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs" / "yolo11n" / "weights" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "eval"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"weights": args.weights}

    # ---- 1. standard metrics (ultralytics val) ----------------------------
    # val() puts the model weights in inference mode on MPS, which breaks any
    # later predict() call ("Inference tensors do not track version counter"),
    # so a dedicated instance is used here and a fresh one for predictions.
    m = YOLO(args.weights).val(data=str(ROOT / "data" / "probe.yaml"), device="mps", verbose=False)
    model = YOLO(args.weights)
    p, r = float(m.box.mp), float(m.box.mr)
    summary["ultralytics_val"] = {
        "precision": round(p, 4), "recall": round(r, 4),
        "f1": round(2 * p * r / (p + r + 1e-9), 4),
        "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
    }

    # ---- 2. per-image predictions, IoU, threshold sweep -------------------
    val_items = load_val_set()
    preds = []  # (image_path, gt, best_conf, best_iou)
    for img_path, gt in val_items:
        res = model.predict(str(img_path), conf=0.01, verbose=False, device="mps")[0]
        if len(res.boxes) == 0:
            preds.append((img_path, gt, 0.0, 0.0))
            continue
        best = int(res.boxes.conf.argmax())
        box = [float(v) for v in res.boxes.xyxy[best]]
        preds.append((img_path, gt, float(res.boxes.conf[best]), iou_xyxy(box, gt)))

    matched = [iou for _, _, c, iou in preds if c >= 0.5 and iou > 0]
    summary["mean_iou_at_conf0.5"] = round(float(np.mean(matched)), 4)

    negatives = []
    for img_path, gt in val_items:
        crop = probe_free_crop(cv2.imread(str(img_path)), gt)
        if crop is not None:
            negatives.append(crop)
    neg_confs = []
    for crop in negatives:
        res = model.predict(crop, conf=0.01, verbose=False, device="mps")[0]
        neg_confs.append(float(res.boxes.conf.max()) if len(res.boxes) else 0.0)
    summary["num_synthetic_negatives"] = len(negatives)

    thresholds = np.arange(0.05, 1.0, 0.05)
    recall_t = [np.mean([(c >= t and iou >= 0.5) for _, _, c, iou in preds]) for t in thresholds]
    fpr_t = [np.mean([c >= t for c in neg_confs]) for t in thresholds]
    # operating point: highest recall with false-alarm rate <= 5%
    ok = [i for i, f in enumerate(fpr_t) if f <= 0.05]
    best_i = ok[int(np.argmax([recall_t[i] for i in ok]))] if ok else int(np.argmin(fpr_t))
    summary["threshold_calibration"] = {
        "chosen_conf_threshold": round(float(thresholds[best_i]), 2),
        "recall_at_threshold": round(float(recall_t[best_i]), 4),
        "false_alarm_rate_at_threshold": round(float(fpr_t[best_i]), 4),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, recall_t, "o-", label="recall on val positives (IoU≥0.5)")
    ax.plot(thresholds, fpr_t, "s-", label="false-alarm rate on synthetic negatives")
    ax.axvline(thresholds[best_i], ls="--", c="gray", label=f"chosen threshold {thresholds[best_i]:.2f}")
    ax.set(xlabel="confidence threshold", ylabel="rate", title="Operating point calibration")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "threshold_calibration.png", dpi=130)

    # ---- 3. runtime benchmark ---------------------------------------------
    bench_imgs = [cv2.imread(str(p)) for p, _ in val_items[:30]]
    runtimes = {}
    for device in ("mps", "cpu"):
        for img in bench_imgs[:5]:  # warm-up
            model.predict(img, verbose=False, device=device)
        times = []
        for img in bench_imgs:
            t0 = time.perf_counter()
            model.predict(img, verbose=False, device=device)
            times.append((time.perf_counter() - t0) * 1000)
        runtimes[device] = {"mean_ms": round(float(np.mean(times)), 1),
                            "std_ms": round(float(np.std(times)), 1),
                            "fps": round(1000 / float(np.mean(times)), 1)}
    summary["runtime_per_image"] = runtimes

    # ---- 4. qualitative grid: 4 best / 4 worst by IoU ----------------------
    ranked = sorted(preds, key=lambda x: x[3])
    picks = ranked[:4] + ranked[-4:]
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, (img_path, gt, conf, iou) in zip(axes.flat, picks):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (int(v) for v in gt)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 40), 2)  # GT orange (img is RGB here)
        res = model.predict(str(img_path), conf=0.25, verbose=False, device="mps")[0]
        if len(res.boxes):
            b = int(res.boxes.conf.argmax())
            px1, py1, px2, py2 = (int(v) for v in res.boxes.xyxy[b])
            cv2.rectangle(img, (px1, py1), (px2, py2), (80, 255, 80), 2)  # pred green
        ax.imshow(img)
        ax.set_title(f"IoU {iou:.2f} conf {conf:.2f}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Worst 4 (top) and best 4 (bottom) val predictions — GT orange, prediction green")
    fig.tight_layout()
    fig.savefig(out / "qualitative_best_worst.png", dpi=130)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
