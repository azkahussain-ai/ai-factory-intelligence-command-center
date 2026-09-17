"""Time-based train/validation/test split and train-only preprocessing stats.

Never shuffles rows randomly - splitting is by simulation day so that no
validation/test-period information can leak backward into training, and no
scaling statistic is ever computed using validation or test rows.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import config


def add_sim_day(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start = pd.Timestamp(config.SIM_START_DATE)
    df["sim_day"] = (pd.to_datetime(df["date"]) - start).dt.days + 1
    return df


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    df = add_sim_day(df)

    train_lo, train_hi = config.TRAIN_DAY_RANGE
    val_lo, val_hi = config.VAL_DAY_RANGE
    test_lo, test_hi = config.TEST_DAY_RANGE

    train = df[(df["sim_day"] >= train_lo) & (df["sim_day"] <= train_hi)].copy()
    val = df[(df["sim_day"] >= val_lo) & (df["sim_day"] <= val_hi)].copy()
    test = df[(df["sim_day"] >= test_lo) & (df["sim_day"] <= test_hi)].copy()

    summary = {
        "train_day_range": list(config.TRAIN_DAY_RANGE),
        "val_day_range": list(config.VAL_DAY_RANGE),
        "test_day_range": list(config.TEST_DAY_RANGE),
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
        "train_failure_rate": float(train["failure_next_shift"].mean()) if len(train) else None,
        "val_failure_rate": float(val["failure_next_shift"].mean()) if len(val) else None,
        "test_failure_rate": float(test["failure_next_shift"].mean()) if len(test) else None,
    }

    # Sanity check: no sim_day should appear in more than one split.
    overlap = (set(train["sim_day"]) & set(val["sim_day"])) | (set(val["sim_day"]) & set(test["sim_day"]))
    summary["day_overlap_between_splits"] = list(overlap)

    return train, val, test, summary


def fit_train_only_scaler(train_df: pd.DataFrame, numeric_cols: list) -> dict:
    """Compute mean/std from training rows only. Applied later (Stage II)
    to validation/test using these same stored stats - never refit on them.
    """
    stats = {}
    for col in numeric_cols:
        if col in train_df.columns:
            stats[col] = {
                "mean": float(train_df[col].mean()),
                "std": float(train_df[col].std() or 1.0),
            }
    return stats


def save_splits(train, val, test, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "validation.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)


def save_scaler_stats(stats: dict, out_path: Path):
    Path(out_path).write_text(json.dumps(stats, indent=2))
