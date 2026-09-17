"""Stage VI — assembles the structured XAI schema and generates the
human-readable summary. All wording is built from the actual explainer
outputs passed in - no per-machine hard-coded text.
"""
from __future__ import annotations


def _format_feature_name(name: str) -> str:
    return name.replace("_", " ")


def build_human_summary(prediction: dict | None, feature_explanations: list,
                         vision: dict | None) -> str:
    parts = []

    if prediction and prediction.get("risk_level"):
        parts.append(
            f"Predicted risk: {prediction['risk_level']} "
            f"(failure probability {prediction.get('probability', 0):.1%})."
        )
    elif prediction is None:
        parts.append("No predictive-maintenance explanation is available for this machine.")

    increasing = [f for f in feature_explanations if f["direction"] == "increases_risk"][:3]
    decreasing = [f for f in feature_explanations if f["direction"] == "decreases_risk"][:2]

    if increasing:
        names = ", ".join(_format_feature_name(f["feature"]) for f in increasing)
        parts.append(f"The prediction is pushed toward failure mainly by: {names}.")
    if decreasing:
        names = ", ".join(_format_feature_name(f["feature"]) for f in decreasing)
        parts.append(f"Pushing toward normal operation: {names}.")

    if vision and vision.get("available"):
        if vision["predicted_class"] == "defect":
            parts.append(
                f"The vision inspection also flagged a defect (confidence {vision['confidence']:.1%}); "
                "see the Grad-CAM overlay for the image region that drove this."
            )
        else:
            parts.append(f"The vision inspection found no defect (confidence {vision['confidence']:.1%}).")

    if not parts:
        return "No explanation could be generated - see the individual component statuses for reasons."
    return " ".join(parts)


def build_xai_result(machine_id: str, pm_explanation: dict, vision_explanation: dict) -> dict:
    """Assemble the Stage VI structured schema from the raw explainer
    outputs. Any component with status != "ok" is preserved as unavailable/
    error rather than dropped or replaced with a fake value.
    """
    prediction = None
    feature_explanations = []
    top_features = []
    tabular_method = None

    if pm_explanation.get("status") == "ok":
        prediction = pm_explanation["prediction"]
        feature_explanations = pm_explanation["feature_explanations"]
        top_features = pm_explanation["top_features"]
        tabular_method = pm_explanation["method"]

    vision_block = {"available": False}
    if vision_explanation.get("status") == "ok":
        vision_block = {
            "available": True,
            "predicted_class": vision_explanation["predicted_class"],
            "confidence": vision_explanation["confidence"],
            "heatmap_path": vision_explanation["heatmap_path"],
            "overlay_path": vision_explanation["overlay_path"],
        }
    elif vision_explanation.get("status") in ("unavailable", "error"):
        vision_block = {"available": False, "reason": vision_explanation.get("reason")}

    human_summary = build_human_summary(
        prediction, feature_explanations, vision_block if vision_block.get("available") else None)

    return {
        "machine_id": machine_id,
        "prediction": prediction or {"status": pm_explanation.get("status"),
                                      "reason": pm_explanation.get("reason")},
        "feature_explanations": feature_explanations,
        "top_features": top_features,
        "vision_explanation": vision_block,
        "human_summary": human_summary,
        "method": {
            "tabular": tabular_method or "unavailable",
            "vision": vision_explanation.get("method", "unavailable") if vision_explanation.get("status") == "ok"
                      else "unavailable",
        },
    }
