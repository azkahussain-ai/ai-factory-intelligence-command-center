"""Stage III - generate a minimal synthetic equipment-image dataset.

No real equipment/product images exist anywhere in the Stage I/II project
(confirmed by inspection - only `reports/figures/*.png` EDA plots exist,
which are not equipment photos). This creates the smallest input needed for
the CV pipeline required by Stage III, clearly documented as synthetic
(see `data/synthetic/SOURCE.md`).

Each image is a 32x32 grayscale render of a machined component (a circular
part against a noisy background). "normal" components are clean; "defect"
components have an added scratch or pit mark. Each image is tagged with a
machine_id from the existing Stage I fleet (round-robin assignment) purely
so Stage III's structured output can be joined with NLP maintenance
predictions later - the pairing itself is synthetic, not a real inspection
record.

Run:
    python -m src.computer_vision.generate_synthetic_images
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IMG_DIR = PROJECT_ROOT / "data" / "synthetic" / "images"
METADATA_PATH = PROJECT_ROOT / "data" / "synthetic" / "images_metadata.csv"

RANDOM_STATE = 42
IMG_SIZE = 32
N_PER_CLASS = 50  # 100 images total - kept small/lightweight by design
MACHINE_FLEET = ["M-01", "M-02", "M-03", "M-04", "M-05", "M-06"]
SPLIT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


def _render_component(rng: np.random.Generator, defect: bool) -> Image.Image:
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=0)
    draw = ImageDraw.Draw(img)

    # Noisy background texture.
    bg = rng.integers(15, 35, size=(IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    img = Image.fromarray(bg, mode="L")
    draw = ImageDraw.Draw(img)

    # Component: a filled circle roughly centered, uniform fill +/- noise.
    r = rng.integers(9, 13)
    cx, cy = IMG_SIZE // 2 + rng.integers(-2, 3), IMG_SIZE // 2 + rng.integers(-2, 3)
    fill_val = int(rng.integers(150, 190))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill_val)

    if defect:
        defect_kind = rng.integers(0, 2)
        if defect_kind == 0:
            # Scratch: a bright thin line across the component.
            x0, y0 = cx - r, cy + rng.integers(-r // 2, r // 2)
            x1, y1 = cx + r, cy + rng.integers(-r // 2, r // 2)
            draw.line([x0, y0, x1, y1], fill=250, width=1)
        else:
            # Pit/spot: a small dark irregular blob near the component center.
            px, py = cx + rng.integers(-r // 2, r // 2), cy + rng.integers(-r // 2, r // 2)
            pr = rng.integers(2, 4)
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=30)

    arr = np.array(img).astype(np.int32)
    noise = rng.integers(-5, 6, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def main():
    rng = np.random.default_rng(RANDOM_STATE)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    idx = 0
    for label, defect in (("normal", False), ("defect", True)):
        for i in range(N_PER_CLASS):
            img = _render_component(rng, defect=defect)
            fname = f"{label}_{i:03d}.png"
            fpath = IMG_DIR / fname
            img.save(fpath)
            records.append({
                "image_path": str(fpath.relative_to(PROJECT_ROOT)),
                "label": label,
                "machine_id": MACHINE_FLEET[idx % len(MACHINE_FLEET)],
            })
            idx += 1

    # Stratified, reproducible split (fit indices per class, shuffled with the same seed).
    rng_split = np.random.default_rng(RANDOM_STATE)
    by_label = {"normal": [], "defect": []}
    for i, r in enumerate(records):
        by_label[r["label"]].append(i)

    split_assignment = {}
    for label, indices in by_label.items():
        indices = np.array(indices)
        rng_split.shuffle(indices)
        n = len(indices)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["validation"])
        for i in indices[:n_train]:
            split_assignment[i] = "train"
        for i in indices[n_train:n_train + n_val]:
            split_assignment[i] = "validation"
        for i in indices[n_train + n_val:]:
            split_assignment[i] = "test"

    for i, r in enumerate(records):
        r["split"] = split_assignment[i]

    with open(METADATA_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "label", "machine_id", "split"])
        writer.writeheader()
        writer.writerows(records)

    counts = {}
    for r in records:
        counts.setdefault(r["split"], {"normal": 0, "defect": 0})
        counts[r["split"]][r["label"]] += 1
    print(f"Generated {len(records)} synthetic images to {IMG_DIR}")
    print(f"Split counts: {counts}")
    print(f"Metadata written to {METADATA_PATH}")


if __name__ == "__main__":
    main()
