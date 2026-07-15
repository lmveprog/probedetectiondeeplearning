"""Self-improvement experiment: temporal-consistency pseudo-labeling.

Question: can the system improve itself from unlabeled flight footage,
without any new human annotation? This simulates a production feedback
loop (drone flies -> mines its own training data -> retrains):

  1. Take the weak model trained on 25% of the training flights
     (runs/lc_25, produced by learning_curve.py).
  2. Run it on the REMAINING training flights, ignoring their human
     labels — they play the role of freshly collected unlabeled footage.
  3. Keep a detection as a pseudo-label only when it is *temporally
     stable*: decent confidence AND consistent (IoU) with the detection
     in an adjacent frame 0.3 s away. Physics is the free supervisor —
     a probe cannot teleport, so isolated firings are discarded.
  4. Retrain the same recipe on 25% human labels + pseudo-labels, and
     evaluate on the untouched validation set.

Since the human labels of the pseudo-labeled flights actually exist, the
script also reports the quality of the auto-generated labels (IoU vs
human), i.e. how trustworthy the feedback loop is.

Usage: python scripts/self_training.py   (~30 min: pseudo-label + retrain)
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train import TRAIN_KWARGS  # noqa: E402
from eda import flight_key  # noqa: E402
from video_demo import timestamp_ms  # noqa: E402

WEAK_WEIGHTS = ROOT / "runs" / "lc_25" / "weights" / "best.pt"
IOU_STABLE = 0.30    # required overlap with a neighboring-frame detection
TARGET_PRECISION = 0.90  # pseudo-label threshold is calibrated for this precision

# NOTE: no absolute confidence threshold here. A model trained on 61 images is
# heavily under-calibrated (best confidences ~0.014 on this run!) even though
# its ranking — hence mAP — is fine. The usable threshold is therefore
# *calibrated on the labeled flights*: the lowest confidence that still yields
# TARGET_PRECISION precision (IoU>=0.5) on data whose labels we do have.


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def labeled_25_flights():
    """Reproduce learning_curve.py's 25% subset (same seed, same order)."""
    by_flight = defaultdict(list)
    for p in sorted((ROOT / "data" / "images" / "train").iterdir()):
        by_flight[flight_key(p.name)].append(p)
    flights = sorted(by_flight)
    random.Random(42).shuffle(flights)
    total = sum(len(v) for v in by_flight.values())
    subset, count = set(), 0
    for fk in flights:
        if count >= 0.25 * total:
            break
        subset.add(fk)
        count += len(by_flight[fk])
    return subset, by_flight


def read_gt(p: Path):
    import cv2
    cx, cy, w, h = [float(v) for v in
                    (ROOT / "data" / "labels" / "train" / (p.stem + ".txt")).read_text().split()[1:5]]
    H, W = cv2.imread(str(p)).shape[:2]
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def calibrate_conf(weak, labeled, by_flight) -> float:
    """Lowest confidence threshold reaching TARGET_PRECISION on the labeled flights."""
    scored = []
    for fk in sorted(labeled):
        for p in by_flight[fk]:
            r = weak.predict(str(p), conf=0.001, verbose=False, device="mps")[0]
            if len(r.boxes):
                b = int(r.boxes.conf.argmax())
                box = [float(v) for v in r.boxes.xyxy[b]]
                scored.append((float(r.boxes.conf[b]), iou_xyxy(box, read_gt(p))))
    scored.sort(reverse=True)
    best_t, hits = None, 0
    for k, (conf, iou) in enumerate(scored, start=1):
        hits += iou >= 0.5
        if hits / k >= TARGET_PRECISION:
            best_t = conf  # lowest threshold (largest k) still meeting the target
    if best_t is None:
        raise SystemExit("weak model never reaches the target precision on labeled data")
    return best_t


def main() -> None:
    if not WEAK_WEIGHTS.exists():
        raise SystemExit("run scripts/learning_curve.py first (needs runs/lc_25)")

    labeled, by_flight = labeled_25_flights()
    unlabeled = {fk: v for fk, v in by_flight.items() if fk not in labeled}
    print(f"labeled flights: {len(labeled)} | unlabeled pool: {len(unlabeled)} flights, "
          f"{sum(len(v) for v in unlabeled.values())} frames")

    conf_min = calibrate_conf(YOLO(str(WEAK_WEIGHTS)), labeled, by_flight)
    print(f"calibrated pseudo-label threshold: {conf_min:.4f} "
          f"(for {TARGET_PRECISION:.0%} precision on the labeled flights)")

    # ---- 1-2. weak model predicts on the unlabeled flights ------------------
    weak = YOLO(str(WEAK_WEIGHTS))
    per_flight = {}  # fk -> [(path, conf, box or None)]
    for fk, paths in sorted(unlabeled.items()):
        paths.sort(key=lambda p: timestamp_ms(p.name))
        dets = []
        for p in paths:
            r = weak.predict(str(p), conf=0.01, verbose=False, device="mps")[0]
            if len(r.boxes):
                b = int(r.boxes.conf.argmax())
                dets.append((p, float(r.boxes.conf[b]), [float(v) for v in r.boxes.xyxy[b]]))
            else:
                dets.append((p, 0.0, None))
        per_flight[fk] = dets

    # ---- 3. temporal-consistency filter -------------------------------------
    pseudo = {}  # path -> box
    rejected = 0
    for fk, dets in per_flight.items():
        for i, (p, conf, box) in enumerate(dets):
            if box is None or conf < conf_min:
                continue
            neighbors = [dets[j] for j in (i - 1, i + 1) if 0 <= j < len(dets)]
            stable = any(nb is not None and iou_xyxy(box, nb) >= IOU_STABLE
                         for _, _, nb in neighbors)
            if stable:
                pseudo[p] = box
            else:
                rejected += 1

    # quality of the pseudo-labels vs the (hidden) human annotations
    import cv2
    ious = []
    for p, box in pseudo.items():
        lbl = ROOT / "data" / "labels" / "train" / (p.stem + ".txt")
        cx, cy, w, h = [float(v) for v in lbl.read_text().split()[1:5]]
        H, W = cv2.imread(str(p)).shape[:2]
        gt = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
        ious.append(iou_xyxy(box, gt))
    coverage = len(pseudo) / sum(len(v) for v in unlabeled.values())
    quality = sum(ious) / len(ious) if ious else 0.0
    print(f"pseudo-labels: {len(pseudo)} kept ({coverage:.0%} of unlabeled frames), "
          f"{rejected} rejected by temporal filter | mean IoU vs human labels: {quality:.3f}")

    # ---- 4. build the self-training set and retrain -------------------------
    import shutil
    img_dir = ROOT / "data" / "images" / "train_self"
    lbl_dir = ROOT / "data" / "labels" / "train_self"
    for d in (img_dir, lbl_dir):  # stale files from a previous run would leak in
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    (ROOT / "data" / "labels" / "train_self.cache").unlink(missing_ok=True)

    def add(p: Path, label_text: str):
        link = img_dir / p.name
        if not link.exists():
            link.symlink_to(p.resolve())
        (lbl_dir / (p.stem + ".txt")).write_text(label_text)

    for fk in labeled:  # human labels for the 25%
        for p in by_flight[fk]:
            add(p, (ROOT / "data" / "labels" / "train" / (p.stem + ".txt")).read_text())
    for p, box in pseudo.items():  # pseudo-labels for the rest
        H, W = cv2.imread(str(p)).shape[:2]
        x1, y1, x2, y2 = box
        add(p, f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")

    yaml = ROOT / "data" / "probe_self.yaml"
    yaml.write_text(f"path: {(ROOT / 'data').resolve()}\ntrain: images/train_self\n"
                    "val: images/val\nnames:\n  0: probe\n")
    YOLO("yolo11n.pt").train(data=str(yaml), device="mps", project=str(ROOT / "runs"),
                             name="self_train", exist_ok=True, **TRAIN_KWARGS)

    # ---- results -------------------------------------------------------------
    rows = {}
    for name, w in [("25% human labels (start)", WEAK_WEIGHTS),
                    ("25% human + auto pseudo-labels", ROOT / "runs" / "self_train" / "weights" / "best.pt"),
                    ("100% human labels (upper bound)", ROOT / "runs" / "yolo11n" / "weights" / "best.pt")]:
        m = YOLO(str(w)).val(data=str(ROOT / "data" / "probe.yaml"), device="mps", verbose=False)
        rows[name] = {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
                      "recall": round(float(m.box.mr), 4)}
        print(f"[self_training] {name}: {rows[name]}", flush=True)

    out = ROOT / "reports" / "self_training"
    out.mkdir(parents=True, exist_ok=True)
    (out / "self_training.json").write_text(json.dumps(
        {"pseudo_labels": len(pseudo), "coverage_of_unlabeled": round(coverage, 4),
         "rejected_by_temporal_filter": rejected,
         "pseudo_label_mean_iou_vs_human": round(quality, 4), "results": rows}, indent=2))
    print(f"[self_training] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
