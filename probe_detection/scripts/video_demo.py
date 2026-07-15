"""Reconstruct flight sequences and render an annotated detection video.

File names embed a capture timestamp (one frame every 300 ms):
<serial>_<flight>_<seq>_1flight_<timestamp_ms>_<cam>.jpg — so each flight can
be replayed as a (sparse) video. This demo runs the detector on every frame
and applies the light temporal smoothing a real onboard pipeline would use:

  - EMA smoothing of the box coordinates (alpha 0.6) while the probe is
    tracked, so the box does not jitter frame to frame;
  - a 2-frame hysteresis before declaring "probe lost", so a single missed
    frame does not flicker the overlay.

By default renders the validation flights (never seen in training) into one
MP4 per flight plus a combined reel, at reports/demo/.

Usage: python scripts/video_demo.py [--weights weights/probe_yolo11n.pt] [--fps 6]
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from eda import flight_key

ROOT = Path(__file__).resolve().parents[1]
GREEN, RED, WHITE = (60, 220, 60), (50, 50, 230), (240, 240, 240)


def timestamp_ms(name: str) -> int:
    return int(re.match(r".*_1flight_(\d+)_\d+\.jpg", name).group(1))


def annotate_frame(img, box, conf, lost, flight, t_ms):
    H, W = img.shape[:2]
    if box is not None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 2)
        label = f"probe {conf:.2f}" if conf > 0 else "probe (tracking)"
        cv2.putText(img, label, (x1, max(18, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)
    elif lost:
        cv2.putText(img, "NO PROBE DETECTED", (10, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    footer = f"{flight}   t={t_ms / 1000:.1f}s"
    cv2.rectangle(img, (0, H - 26), (W, H), (0, 0, 0), -1)
    cv2.putText(img, footer, (10, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights" / "probe_yolo11n.pt"))
    ap.add_argument("--split", default="val", choices=["val", "train", "all"],
                    help="'val' (default) shows honest never-seen flights; 'all' includes training flights")
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default=None, help="cpu, mps, cuda:0; auto if omitted")
    ap.add_argument("--out", default=str(ROOT / "reports" / "demo"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    flights = defaultdict(list)
    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        for p in (ROOT / "data" / "images" / split).iterdir():
            flights[flight_key(p.name)].append(p)

    all_frames = []
    for flight, paths in sorted(flights.items()):
        paths.sort(key=lambda p: timestamp_ms(p.name))
        writer = None
        ema_box, misses = None, 99
        for p in paths:
            img = cv2.imread(str(p))
            res = model.predict(img, conf=args.conf, verbose=False, device=args.device)[0]

            if len(res.boxes):
                best = int(res.boxes.conf.argmax())
                box = np.array([float(v) for v in res.boxes.xyxy[best]])
                conf = float(res.boxes.conf[best])
                ema_box = box if ema_box is None or misses >= 2 else 0.6 * box + 0.4 * ema_box
                misses = 0
            else:
                misses += 1
                conf = 0.0
                if misses >= 2:  # hysteresis: declare lost after 2 consecutive misses
                    ema_box = None

            frame = annotate_frame(img.copy(), ema_box if misses < 2 else None,
                                   conf, misses >= 2, flight, timestamp_ms(p.name))
            if writer is None:
                H, W = frame.shape[:2]
                writer = cv2.VideoWriter(str(out / f"{flight}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
            writer.write(frame)
            all_frames.append(frame)
        writer.release()
        print(f"{flight}: {len(paths)} frames -> {out / f'{flight}.mp4'}")

    H, W = all_frames[0].shape[:2]
    reel = cv2.VideoWriter(str(out / "demo_reel.mp4"),
                           cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in all_frames:
        reel.write(f)
    reel.release()
    print(f"combined reel ({len(all_frames)} frames) -> {out / 'demo_reel.mp4'}")


if __name__ == "__main__":
    main()
