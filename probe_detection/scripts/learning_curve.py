"""Data-scaling study: how much would more annotated data buy?

Trains the exact same recipe (train.TRAIN_KWARGS) on nested subsets of the
training flights (25% / 50% / 75% / 100% of training images, whole flights
only, same fixed validation set) and plots mAP against dataset size. The
slope at 100% answers a practical question: is collecting more data still
worth it, and roughly how much per annotated image?

Subsets are nested (the 25% flights are contained in the 50% ones, etc.),
which is the standard protocol for learning curves — differences between
points then reflect added data, not a different data mix. The 100% point
reuses the existing baseline run when available (identical recipe and seed).

Usage: python scripts/learning_curve.py   (~45 min on Apple Silicon)
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from train import TRAIN_KWARGS  # noqa: E402
from eda import flight_key  # noqa: E402

FRACTIONS = [0.25, 0.5, 0.75, 1.0]
BASELINE_100 = ROOT / "runs" / "yolo11n" / "weights" / "best.pt"


def main() -> None:
    by_flight = defaultdict(list)
    for p in sorted((ROOT / "data" / "images" / "train").iterdir()):
        by_flight[flight_key(p.name)].append(p)
    flights = sorted(by_flight)
    random.Random(42).shuffle(flights)
    total = sum(len(v) for v in by_flight.values())

    results = []
    for frac in FRACTIONS:
        pct = int(frac * 100)
        subset, nfl = [], 0
        for fk in flights:  # same flight order for every fraction -> nested subsets
            if len(subset) >= frac * total:
                break
            subset.extend(by_flight[fk])
            nfl += 1

        if frac == 1.0 and BASELINE_100.exists():
            best = BASELINE_100  # same recipe, same seed, same data: reuse
        else:
            list_file = ROOT / "data" / f"train_{pct}.txt"
            # keep the symlink paths (data/images/train/...) — resolving them would
            # point at the original dataset folder, where ultralytics finds no labels
            list_file.write_text("\n".join(str(p) for p in subset) + "\n")
            yaml = ROOT / "data" / f"probe_{pct}.yaml"
            yaml.write_text(
                f"path: {(ROOT / 'data').resolve()}\n"
                f"train: train_{pct}.txt\n"
                "val: images/val\n"
                "names:\n  0: probe\n"
            )
            YOLO("yolo11n.pt").train(
                data=str(yaml), device="mps", project=str(ROOT / "runs"),
                name=f"lc_{pct}", exist_ok=True, **TRAIN_KWARGS,
            )
            best = ROOT / "runs" / f"lc_{pct}" / "weights" / "best.pt"

        m = YOLO(str(best)).val(data=str(ROOT / "data" / "probe.yaml"), device="mps", verbose=False)
        results.append({"fraction": frac, "images": len(subset), "flights": nfl,
                        "mAP50": round(float(m.box.map50), 4),
                        "mAP50_95": round(float(m.box.map), 4),
                        "recall": round(float(m.box.mr), 4)})
        print(f"[learning_curve] {pct}%: {results[-1]}", flush=True)

    out = ROOT / "reports" / "learning_curve"
    out.mkdir(parents=True, exist_ok=True)
    (out / "learning_curve.json").write_text(json.dumps(results, indent=2))

    xs = [r["images"] for r in results]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(xs, [r["mAP50"] for r in results], "o-", label="mAP@0.5")
    ax.plot(xs, [r["mAP50_95"] for r in results], "s-", label="mAP@0.5:0.95")
    ax.plot(xs, [r["recall"] for r in results], "^-", label="recall")
    for r in results:
        ax.annotate(f"{r['flights']} flights", (r["images"], r["mAP50"]),
                    textcoords="offset points", xytext=(0, 8), fontsize=8, ha="center")
    ax.set(xlabel="training images (whole flights)", ylabel="metric on fixed val set",
           title="Learning curve — is more data worth collecting?", ylim=(0, 1))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_curve.png", dpi=130)
    print(f"[learning_curve] saved to {out}", flush=True)


if __name__ == "__main__":
    main()
