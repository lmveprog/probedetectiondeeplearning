"""Probe detection inference script.

Iterates over every image of a folder, detects the ultrasonic thickness
probe if present, draws its bounding box and either saves the annotated
image to an output folder (default) or shows it in a window (--show).
When no probe is detected the image is marked "NO PROBE DETECTED" and the
absence is also reported on the console.

Examples:
  python inference.py path/to/images
  python inference.py path/to/images --output results --conf 0.4
  python inference.py path/to/images --show

The detection result for each image is also written to <output>/detections.json:
  {"image.jpg": {"detected": true, "bbox_xyxy": [x1, y1, x2, y2], "confidence": 0.93}}
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "weights" / "probe_yolo11n.pt"

BOX_COLOR = (60, 220, 60)      # green (BGR)
ALERT_COLOR = (50, 50, 230)    # red   (BGR)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_folder", help="folder containing the images to process")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="trained model weights (.pt)")
    # 0.30 = operating point calibrated in evaluate.py (max recall at <=5% false-alarm rate)
    ap.add_argument("--conf", type=float, default=0.3, help="confidence threshold below which 'no probe' is reported")
    ap.add_argument("--output", default="output", help="folder where annotated images are saved")
    ap.add_argument("--show", action="store_true", help="show each image in a window instead of saving")
    ap.add_argument("--device", default=None, help="inference device (e.g. cpu, mps, cuda:0); auto if omitted")
    return ap.parse_args()


def annotate(img, result, conf_thres: float):
    """Draw the best probe detection on img; return (annotated, detection dict)."""
    boxes = result.boxes
    if len(boxes) > 0:
        best = int(boxes.conf.argmax())
        conf = float(boxes.conf[best])
        if conf >= conf_thres:
            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[best])
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, 2)
            label = f"probe {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ty = y1 - 6 if y1 - th - 10 > 0 else y2 + th + 8
            cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 4, ty + 4), BOX_COLOR, -1)
            cv2.putText(img, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            return img, {"detected": True, "bbox_xyxy": [x1, y1, x2, y2], "confidence": round(conf, 4)}

    cv2.putText(img, "NO PROBE DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ALERT_COLOR, 2)
    return img, {"detected": False}


def main() -> None:
    args = parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        raise SystemExit(f"error: {input_folder} is not a folder")
    image_paths = sorted(p for p in input_folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"error: no image files found in {input_folder}")

    model = YOLO(args.weights)
    output = Path(args.output)
    if not args.show:
        output.mkdir(parents=True, exist_ok=True)

    detections, times = {}, []
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[skip] {path.name}: unreadable image")
            continue

        t0 = time.perf_counter()
        # conf=0.05 so we can apply our own calibrated threshold in annotate()
        result = model.predict(img, conf=0.05, verbose=False, device=args.device)[0]
        times.append(time.perf_counter() - t0)

        img, det = annotate(img, result, args.conf)
        detections[path.name] = det
        status = (f"probe at {det['bbox_xyxy']} (conf {det['confidence']:.2f})"
                  if det["detected"] else "NO PROBE DETECTED")
        print(f"{path.name}: {status}")

        if args.show:
            cv2.imshow("probe detection (press any key, q to quit)", img)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break
        else:
            cv2.imwrite(str(output / path.name), img)

    if args.show:
        cv2.destroyAllWindows()
    else:
        (output / "detections.json").write_text(json.dumps(detections, indent=2))
        print(f"\nannotated images and detections.json saved to {output}/")

    n_det = sum(d["detected"] for d in detections.values())
    avg_ms = 1000 * sum(times) / max(len(times), 1)
    print(f"\n{len(detections)} images processed | probe detected in {n_det} | "
          f"avg inference {avg_ms:.1f} ms/image (first image includes model warm-up)")


if __name__ == "__main__":
    main()
