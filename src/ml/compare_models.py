"""Stage II - Combine baseline ML and GRU metrics into one comparison report.

Run after train_baseline.py and train_gru.py:
    python -m src.ml.compare_models
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def main():
    with open(REPORTS_DIR / "stage2_baseline_metrics.json") as f:
        baseline = json.load(f)
    with open(REPORTS_DIR / "stage2_gru_metrics.json") as f:
        gru = json.load(f)

    all_models = {k: v for k, v in baseline.items() if k != "best_model"}
    all_models["gru"] = gru

    summary = {}
    for name, res in all_models.items():
        v = res["validation"]
        t = res["test"]
        summary[name] = {
            "validation": {"precision": v["precision"], "recall": v["recall"],
                            "f1": v["f1"], "roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"]},
            "test": {"precision": t["precision"], "recall": t["recall"],
                      "f1": t["f1"], "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"]},
        }

    best_overall = max(summary, key=lambda k: summary[k]["validation"]["f1"])

    report = {
        "models_compared": list(summary.keys()),
        "metric_summary": summary,
        "best_model_by_validation_f1": best_overall,
        "note": (
            "Positive class (failure_next_shift) is extremely rare "
            f"(train=7/900, val=3/270, test=6/444). ROC-AUC/PR-AUC and any "
            "near-perfect scores must be read with this in mind - they reflect "
            "a handful of positive examples separated by a strong synthetic "
            "precursor signal (near-total downtime_ratio in the shift before "
            "failure), not validated generalization to novel failure modes."
        ),
    }

    out_path = REPORTS_DIR / "stage2_model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Best model overall (validation F1): {best_overall}")
    for name, s in summary.items():
        print(f"  {name:20s} val F1={s['validation']['f1']:.3f}  "
              f"test F1={s['test']['f1']:.3f}  test ROC-AUC={s['test']['roc_auc']}")
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
