"""Stage VI — Grad-CAM for Stage III's computer-vision CNN.

`SimpleCNNBinaryClassifier` (src/computer_vision/cnn_numpy.py) is a
hand-rolled NumPy CNN with no autodiff framework, so no off-the-shelf
Grad-CAM library applies. Grad-CAM is implemented directly here by manually
back-propagating the gradient of the predicted class logit to the network's
one convolutional layer's activations (its single conv layer *is* the
"last conv layer" - there is only one), reusing the exact forward-pass
cache the model already produces and mirroring the chain-rule steps the
model's own `backward()` method uses (just with a constant seed gradient of
1.0 at the logit instead of a label-dependent loss gradient, since Grad-CAM
explains the logit itself, not a loss).

No changes are made to `cnn_numpy.py` - this module only reads its public
`forward()` output and cached intermediate values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.config import PROJECT_ROOT, REPORTS_DIR
from src.computer_vision.cnn_numpy import SimpleCNNBinaryClassifier
from src.agents.schemas import unavailable_result, error_result

MODELS_DIR = PROJECT_ROOT / "models"
XAI_OUTPUT_DIR = REPORTS_DIR / "xai"


def _gradient_wrt_relu_out(model: SimpleCNNBinaryClassifier, cache: dict) -> np.ndarray:
    """d(logit)/d(relu_out), for a single sample - the same chain of steps
    as `SimpleCNNBinaryClassifier.backward()`, but seeded with d(logit)=1
    instead of a label-dependent BCE gradient, since Grad-CAM explains the
    raw class score, not a training loss.
    """
    dlogits = np.array([[1.0]])
    da1 = dlogits @ model.W2.T
    dz1 = da1 * (cache["z1"] > 0)
    dflat = dz1 @ model.W1.T

    ph, pw = cache["ph"], cache["pw"]
    dpooled = dflat.reshape(1, ph, pw, model.n_filters)

    drelu_cropped = np.zeros((1, ph, 2, pw, 2, model.n_filters))
    argmax = cache["pool_argmax"]
    idx_n, idx_ph, idx_pw, idx_f = np.meshgrid(
        np.arange(1), np.arange(ph), np.arange(pw), np.arange(model.n_filters), indexing="ij")
    row = argmax // 2
    col = argmax % 2
    drelu_cropped[idx_n, idx_ph, row, idx_pw, col, idx_f] = dpooled
    drelu_cropped = drelu_cropped.reshape(1, ph * 2, pw * 2, model.n_filters)

    oh, ow = cache["oh"], cache["ow"]
    drelu = np.zeros((1, oh, ow, model.n_filters))
    drelu[:, :ph * 2, :pw * 2, :] = drelu_cropped
    return drelu[0]  # (oh, ow, n_filters)


def _upsample_nearest(arr: np.ndarray, target_size: int) -> np.ndarray:
    factor = target_size / arr.shape[0]
    idx = (np.arange(target_size) / factor).astype(int).clip(0, arr.shape[0] - 1)
    return arr[idx][:, idx]


def _explain_array(model: SimpleCNNBinaryClassifier, img: np.ndarray) -> dict:
    """Core Grad-CAM computation for an already-loaded model and a single
    (H, W) float64 [0,1] grayscale image array. Used by both `explain()`
    (Stage VI, stored Stage III images) and the Stage IX web app's live
    image-upload inference - one implementation, two callers, no duplicated
    pipeline.
    """
    X = img[np.newaxis, ...]
    probs, cache = model.forward(X)
    confidence = float(probs[0])

    gradient = _gradient_wrt_relu_out(model, cache)
    alpha = gradient.mean(axis=(0, 1))
    feature_maps = cache["relu_out"][0]
    cam = np.maximum((alpha * feature_maps).sum(axis=-1), 0)
    cam_max = cam.max()
    cam_norm = cam / cam_max if cam_max > 1e-8 else cam
    heatmap = _upsample_nearest(cam_norm, model.img_size)
    predicted_class = "defect" if confidence >= 0.5 else "normal"
    return {"confidence": confidence, "predicted_class": predicted_class, "heatmap": heatmap, "image": img}


def explain_uploaded_image(img: np.ndarray) -> dict:
    """Live Grad-CAM inference on an arbitrary uploaded image (Stage IX),
    reusing the exact same trained Stage III model and Grad-CAM math as
    `explain()` - not a second model, not a second pipeline.
    """
    model_path = MODELS_DIR / "cv_cnn_model.npz"
    if not model_path.exists():
        return unavailable_result("Stage III CV model file not found")
    try:
        model = SimpleCNNBinaryClassifier.load(model_path)
        if img.shape != (model.img_size, model.img_size):
            img_pil = Image.fromarray((img * 255).astype(np.uint8)).resize(
                (model.img_size, model.img_size))
            img = np.array(img_pil, dtype=np.float64) / 255.0
        result = _explain_array(model, img)
    except Exception as exc:
        return error_result(f"Grad-CAM failed for uploaded image: {exc}")

    XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay_path = XAI_OUTPUT_DIR / "uploaded_image_gradcam_overlay.png"
    _save_overlay(result["image"], result["heatmap"], overlay_path)
    return {
        "status": "ok",
        "available": True,
        "predicted_class": result["predicted_class"],
        "confidence": round(result["confidence"], 4),
        "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
        "method": "Grad-CAM (same Stage III model + Stage VI method, live inference on the uploaded image)",
    }


def explain(machine_id: str) -> dict:
    """Grad-CAM for the machine's most recent CV inspection image, reusing
    Stage III's existing structured output to identify which image and
    prediction to explain (same source Stage V's Vision Agent reads).
    """
    from src.agents.vision_agent import _load_stage3_record  # reuse, not duplicate

    record = _load_stage3_record(machine_id)
    if record is None or "cv_source_image" not in record:
        return unavailable_result(f"No Stage III CV result/image found for machine {machine_id}")

    model_path = MODELS_DIR / "cv_cnn_model.npz"
    if not model_path.exists():
        return unavailable_result("Stage III CV model file not found")

    image_path = PROJECT_ROOT / record["cv_source_image"]
    if not image_path.exists():
        return unavailable_result(f"Source image not found on disk: {record['cv_source_image']}")

    try:
        model = SimpleCNNBinaryClassifier.load(model_path)
        img = np.array(Image.open(image_path), dtype=np.float64) / 255.0
        result = _explain_array(model, img)
        confidence = result["confidence"]
        heatmap = result["heatmap"]
    except Exception as exc:
        return error_result(f"Grad-CAM failed for {machine_id}: {exc}")

    XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heatmap_path = XAI_OUTPUT_DIR / f"{machine_id}_gradcam_heatmap.png"
    overlay_path = XAI_OUTPUT_DIR / f"{machine_id}_gradcam_overlay.png"
    _save_heatmap(heatmap, heatmap_path)
    _save_overlay(img, heatmap, overlay_path)

    predicted_class = result["predicted_class"]
    return {
        "status": "ok",
        "machine_id": machine_id,
        "available": True,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "source_image": record["cv_source_image"],
        "heatmap_path": str(heatmap_path.relative_to(PROJECT_ROOT)),
        "overlay_path": str(overlay_path.relative_to(PROJECT_ROOT)),
        "method": "Grad-CAM (manual backprop to the model's single conv layer - "
                  "no autodiff framework available for this hand-rolled NumPy CNN)",
    }


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """Simple red-intensity colormap (no matplotlib dependency needed here)."""
    r = (heatmap * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1 - heatmap) * 80).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _save_heatmap(heatmap: np.ndarray, path: Path):
    Image.fromarray(_colorize(heatmap)).save(path)


def _save_overlay(original_gray: np.ndarray, heatmap: np.ndarray, path: Path, alpha: float = 0.45):
    base_rgb = np.stack([original_gray] * 3, axis=-1)
    color_heat = _colorize(heatmap).astype(np.float64) / 255.0
    overlay = (1 - alpha) * base_rgb + alpha * color_heat
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(path)
