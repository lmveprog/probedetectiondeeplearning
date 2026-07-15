"""Fine-tune a COCO-pretrained YOLO11 model on the probe dataset.

Hyperparameter choices (documented for the report):
  - imgsz=640: native image width; YOLO letterboxes 640x400 -> 640x640.
  - epochs=150 with patience=40: small dataset, cheap epochs; early stopping
    picks the best checkpoint on val mAP.
  - batch=16, optimizer/lr: ultralytics auto settings (AdamW, lr auto-scaled).
  - hsv_h=0, hsv_s=0: the nav-cam images are grayscale, hue/saturation jitter
    is a no-op at best; hsv_v=0.4 keeps brightness jitter, which matches the
    real variability (onboard LED lighting, overexposed probe silhouettes).
  - flipud=0.5: the probe is physically mounted either on top or below the
    drone, so vertical flips are a realistic augmentation for this task.
  - degrees=10, translate/scale defaults, mosaic until the last 15 epochs.

Usage: python train.py [--model yolo11n.pt] [--epochs 150]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# Single source of truth for the training recipe — scripts/learning_curve.py
# imports this so every data-fraction run uses the exact same settings.
TRAIN_KWARGS = dict(
    epochs=150,
    patience=40,
    batch=16,
    imgsz=640,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.4,
    degrees=10.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=1.0,
    close_mosaic=15,
    seed=42,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--data", default=str(ROOT / "data" / "probe.yaml"))
    ap.add_argument("--name", default=None, help="run name (defaults to the model stem)")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name or Path(args.model).stem,
        exist_ok=True,
        **{**TRAIN_KWARGS, "epochs": args.epochs, "batch": args.batch},
    )


if __name__ == "__main__":
    main()
