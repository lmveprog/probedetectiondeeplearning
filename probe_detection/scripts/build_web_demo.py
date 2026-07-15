"""Build a fully self-contained, zero-install browser demo.

Packs everything into ONE html file (web_demo/probe_demo.html):
  - the trained detector exported to ONNX (base64),
  - the onnxruntime-web WASM runtime (base64) and its JS loader (inlined),
  - a sample validation image (base64),
  - ~150 lines of JS: letterbox preprocessing, inference, box decoding.

Anyone can double-click the file and drop their own images on it — the
actual trained network runs locally in the browser (WebAssembly + SIMD),
no Python, no install, no upload. This also demonstrates the model runs
in an environment far more constrained than a Jetson.

Usage: python scripts/build_web_demo.py   (downloads the ORT runtime once)
"""

import base64
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORT_VERSION = "1.14.0"  # last line with a non-threaded SIMD wasm + opset<=17 support
CDN = f"https://cdn.jsdelivr.net/npm/onnxruntime-web@{ORT_VERSION}/dist"
CACHE = ROOT / "web_demo" / ".ort_cache"

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Probe Detection — In-Browser Demo</title>
<style>
  body { font: 15px/1.5 -apple-system, "Segoe UI", sans-serif; background: #0f172a;
         color: #e2e8f0; max-width: 860px; margin: 2em auto; padding: 0 1em; }
  h1 { font-size: 1.4em; } .sub { color: #94a3b8; margin-bottom: 1.2em; }
  #drop { border: 2px dashed #475569; border-radius: 10px; padding: 2.2em;
          text-align: center; cursor: pointer; transition: border-color .2s; }
  #drop.hover { border-color: #38bdf8; }
  #status { margin: .8em 0; color: #94a3b8; min-height: 1.4em; }
  canvas { max-width: 100%; border-radius: 8px; display: block; margin-top: .6em; }
  .row { display: flex; gap: 1.2em; align-items: center; margin-top: .8em; flex-wrap: wrap; }
  button { background: #1d4ed8; color: white; border: 0; border-radius: 6px;
           padding: .55em 1.1em; font-size: 1em; cursor: pointer; }
  button:disabled { opacity: .4; cursor: default; }
  label { color: #94a3b8; }
  input[type=range] { vertical-align: middle; }
</style>
</head>
<body>
<h1>Probe Detection — In-Browser Demo</h1>
<p class="sub">The fine-tuned YOLO11n runs locally in your browser (WebAssembly).
Nothing is installed or uploaded — drop any image below.</p>

<div id="drop">Drop an image here, or click to choose a file</div>
<input type="file" id="file" accept="image/*" hidden>
<div class="row">
  <button id="sample" disabled>Try a sample image</button>
  <label>Confidence threshold
    <input type="range" id="thr" min="0.05" max="0.9" step="0.05" value="0.30">
    <span id="thrv">0.30</span></label>
</div>
<div id="status">Loading model…</div>
<canvas id="cv" hidden></canvas>

<script>__ORT_JS__</script>
<script>
"use strict";
const WASM_B64 = "__WASM_B64__";
const MODEL_B64 = "__MODEL_B64__";
const SAMPLE_B64 = "__SAMPLE_B64__";
const SIZE = 640;

const $ = id => document.getElementById(id);
const status = m => $("status").textContent = m;

function b64bytes(b64) {
  const bin = atob(b64), a = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
  return a;
}

let session = null, lastImage = null;

async function init() {
  const wasmURL = URL.createObjectURL(new Blob([b64bytes(WASM_B64)], {type: "application/wasm"}));
  ort.env.wasm.numThreads = 1;               // threaded build not embedded
  ort.env.wasm.wasmPaths = {"ort-wasm-simd.wasm": wasmURL, "ort-wasm.wasm": wasmURL};
  session = await ort.InferenceSession.create(b64bytes(MODEL_B64).buffer,
                                              {executionProviders: ["wasm"]});
  $("sample").disabled = false;
  status("Model ready — drop an image.");
  if (location.hash === "#autorun") runSample();
}

function preprocess(img) {
  const r = Math.min(SIZE / img.width, SIZE / img.height);
  const nw = Math.round(img.width * r), nh = Math.round(img.height * r);
  const padX = (SIZE - nw) / 2, padY = (SIZE - nh) / 2;
  const off = document.createElement("canvas");
  off.width = off.height = SIZE;
  const ctx = off.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, SIZE, SIZE);
  ctx.drawImage(img, padX, padY, nw, nh);
  const d = ctx.getImageData(0, 0, SIZE, SIZE).data;
  const chw = new Float32Array(3 * SIZE * SIZE);
  for (let p = 0; p < SIZE * SIZE; p++)
    for (let c = 0; c < 3; c++)
      chw[c * SIZE * SIZE + p] = d[p * 4 + c] / 255;
  return {tensor: new ort.Tensor("float32", chw, [1, 3, SIZE, SIZE]), r, padX, padY};
}

async function detect(img) {
  const t0 = performance.now();
  const {tensor, r, padX, padY} = preprocess(img);
  const out = await session.run({[session.inputNames[0]]: tensor});
  const o = out[session.outputNames[0]].data;   // [1, 5, 8400] -> ch*8400 + i
  const N = 8400;
  let best = 0, conf = 0;
  for (let i = 0; i < N; i++) if (o[4 * N + i] > conf) { conf = o[4 * N + i]; best = i; }
  const cx = o[best], cy = o[N + best], w = o[2 * N + best], h = o[3 * N + best];
  const box = [(cx - w / 2 - padX) / r, (cy - h / 2 - padY) / r,
               (cx + w / 2 - padX) / r, (cy + h / 2 - padY) / r]
              .map((v, i) => Math.max(0, Math.min(v, i % 2 ? img.height : img.width)));
  return {conf, box, ms: performance.now() - t0};
}

function draw(img, det, thr) {
  const cv = $("cv"), ctx = cv.getContext("2d");
  cv.width = img.width; cv.height = img.height; cv.hidden = false;
  ctx.drawImage(img, 0, 0);
  const s = Math.max(1.5, img.width / 400);
  if (det.conf >= thr) {
    const [x1, y1, x2, y2] = det.box;
    ctx.strokeStyle = "#4ade80"; ctx.lineWidth = 2 * s;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.font = `bold ${13 * s}px sans-serif`;
    const label = `probe ${det.conf.toFixed(2)}`;
    ctx.fillStyle = "#4ade80";
    ctx.fillRect(x1, Math.max(0, y1 - 18 * s), ctx.measureText(label).width + 8 * s, 18 * s);
    ctx.fillStyle = "#052e16";
    ctx.fillText(label, x1 + 4 * s, Math.max(13 * s, y1 - 5 * s));
    status(`Probe detected — confidence ${det.conf.toFixed(2)} — ${det.ms.toFixed(0)} ms`);
  } else {
    ctx.font = `bold ${16 * s}px sans-serif`;
    ctx.fillStyle = "#f87171";
    ctx.fillText("NO PROBE DETECTED", 12, 24 * s);
    status(`No probe detected (best candidate ${det.conf.toFixed(2)} below threshold) — ${det.ms.toFixed(0)} ms`);
  }
}

async function runOn(img) {
  lastImage = img;
  status("Running…");
  await new Promise(r => setTimeout(r, 20));   // let the status paint
  draw(img, await detect(img), parseFloat($("thr").value));
}

function loadFile(file) {
  const img = new Image();
  img.onload = () => runOn(img);
  img.src = URL.createObjectURL(file);
}

function runSample() {
  const img = new Image();
  img.onload = () => runOn(img);
  img.src = "data:image/jpeg;base64," + SAMPLE_B64;
}

$("drop").onclick = () => $("file").click();
$("file").onchange = e => e.target.files[0] && loadFile(e.target.files[0]);
$("drop").ondragover = e => { e.preventDefault(); $("drop").classList.add("hover"); };
$("drop").ondragleave = () => $("drop").classList.remove("hover");
$("drop").ondrop = e => {
  e.preventDefault(); $("drop").classList.remove("hover");
  e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]);
};
$("sample").onclick = runSample;
$("thr").oninput = () => {
  $("thrv").textContent = parseFloat($("thr").value).toFixed(2);
  if (lastImage) runOn(lastImage);
};

init().catch(e => status("Failed to load model: " + e));
</script>
</body>
</html>
"""


def fetch(name: str) -> bytes:
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / name
    if not cached.exists():
        print(f"downloading {name}…")
        urllib.request.urlretrieve(f"{CDN}/{name}", cached)
    return cached.read_bytes()


def main() -> None:
    onnx = ROOT / "weights" / "probe_yolo11n.onnx"
    if not onnx.exists():
        raise SystemExit("run: YOLO('weights/probe_yolo11n.pt').export(format='onnx', imgsz=640, opset=12)")

    ort_js = fetch("ort.min.js").decode("utf-8")
    if "</script>" in ort_js:  # would break the inline <script> block
        ort_js = ort_js.replace("</script>", "<\\/script>")
    wasm = fetch("ort-wasm-simd.wasm")

    sample = sorted((ROOT / "data" / "images" / "val").iterdir())[0].resolve()

    html = (TEMPLATE
            .replace("__ORT_JS__", ort_js)
            .replace("__WASM_B64__", base64.b64encode(wasm).decode())
            .replace("__MODEL_B64__", base64.b64encode(onnx.read_bytes()).decode())
            .replace("__SAMPLE_B64__", base64.b64encode(sample.read_bytes()).decode()))

    out = ROOT / "web_demo" / "probe_demo.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"built {out} ({out.stat().st_size / 1e6:.1f} MB, sample: {sample.name})")


if __name__ == "__main__":
    main()
