"""Export the trained model to ONNX and benchmark deployment options.

On a Jetson the production path is ONNX -> TensorRT (FP16/INT8), which
cannot be built on a Mac. This script demonstrates the exact same first
step of that pipeline and measures what is measurable here:

  - PyTorch CPU baseline
  - ONNX Runtime CPU (fp32)     <- the graph TensorRT would consume
  - ONNX Runtime CPU (int8, dynamic quantization) + model size reduction

The int8 latency on CPU is only indicative (CPU int8 kernels differ from
Jetson tensor cores), but the size reduction and the unchanged-detection
sanity check carry over directly.

Usage: python scripts/export_bench.py [--weights weights/probe_yolo11n.pt]
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def letterbox_batch(img, size=640):
    """Minimal YOLO preprocessing: letterbox to size x size, NCHW float32 [0,1]."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def bench(fn, inputs, warmup=5, reps=30):
    for x in inputs[:warmup]:
        fn(x)
    times = []
    for x in inputs[:reps]:
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.mean(times)), 1), round(float(np.std(times)), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "weights" / "probe_yolo11n.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    val_imgs = [cv2.imread(str(p)) for p in sorted((ROOT / "data" / "images" / "val").iterdir())[:30]]
    tensors = [letterbox_batch(im) for im in val_imgs]
    results = {}

    # ---- PyTorch CPU baseline ---------------------------------------------
    model = YOLO(args.weights)
    mean, std = bench(lambda im: model.predict(im, verbose=False, device="cpu"), val_imgs)
    results["pytorch_cpu"] = {"mean_ms": mean, "std_ms": std,
                              "size_mb": round(Path(args.weights).stat().st_size / 1e6, 1)}

    # ---- ONNX export (the graph a Jetson TensorRT engine is built from) ----
    onnx_path = Path(YOLO(args.weights).export(format="onnx", imgsz=640, verbose=False))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    mean, std = bench(lambda x: sess.run(None, {iname: x}), tensors)
    results["onnxruntime_cpu_fp32"] = {"mean_ms": mean, "std_ms": std,
                                       "size_mb": round(onnx_path.stat().st_size / 1e6, 1)}

    # ---- int8 dynamic quantization -----------------------------------------
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q_path = onnx_path.with_stem(onnx_path.stem + "_int8")
    quantize_dynamic(str(onnx_path), str(q_path), weight_type=QuantType.QUInt8)
    qsess = ort.InferenceSession(str(q_path), providers=["CPUExecutionProvider"])
    mean, std = bench(lambda x: qsess.run(None, {iname: x}), tensors)
    results["onnxruntime_cpu_int8"] = {"mean_ms": mean, "std_ms": std,
                                       "size_mb": round(q_path.stat().st_size / 1e6, 1)}

    # ---- sanity check: fp32 vs int8 agreement on one image -----------------
    ref = sess.run(None, {iname: tensors[0]})[0]
    qout = qsess.run(None, {iname: tensors[0]})[0]
    best_ref, best_q = ref[0, 4].argmax(), qout[0, 4].argmax()
    results["int8_sanity"] = {
        "fp32_best_conf": round(float(ref[0, 4].max()), 3),
        "int8_best_conf": round(float(qout[0, 4].max()), 3),
        "center_shift_px": round(float(np.abs(ref[0, :2, best_ref] - qout[0, :2, best_q]).max()), 1),
    }

    out = ROOT / "reports" / "export_benchmark.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
