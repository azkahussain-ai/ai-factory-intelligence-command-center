"""Exploratory data analysis for the integrated Stage I dataset.

Every number in the resulting summary is computed from the actual
dataframes passed in - nothing here is a hard-coded finding. Plots are
limited to a handful that each answer one specific analytical question,
per the Stage I scope (no plot-dump).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 3),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def numeric_summary(df: pd.DataFrame, cols: list) -> dict:
    present = [c for c in cols if c in df.columns]
    return df[present].describe().round(3).to_dict()


def failure_rate_by_machine(df: pd.DataFrame, target_col: str) -> dict:
    return df.groupby("machine_id")[target_col].mean().round(4).to_dict()


def correlation_matrix(df: pd.DataFrame, cols: list) -> dict:
    present = [c for c in cols if c in df.columns]
    return df[present].corr(numeric_only=True).round(3).to_dict()


def generate_plots(integrated_df: pd.DataFrame, sensor_df: pd.DataFrame, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Failure rate by machine - which machines are riskiest.
    fig, ax = plt.subplots(figsize=(6, 4))
    rates = integrated_df.groupby("machine_id")["failure_occurred_this_shift"].mean()
    rates.plot(kind="bar", ax=ax, color="#c0392b")
    ax.set_ylabel("Failure rate (per shift)")
    ax.set_title("Failure Rate by Machine")
    fig.tight_layout()
    fig.savefig(out_dir / "failure_rate_by_machine.png", dpi=100)
    plt.close(fig)

    # 2. Vibration trend over time for one degrading machine (illustrates
    # the ramp-to-failure pattern the DL model will need to learn later).
    fig, ax = plt.subplots(figsize=(8, 4))
    for machine_id, g in sensor_df.groupby("machine_id"):
        ax.plot(g["timestamp"], g["vibration_mm_s"], label=machine_id, alpha=0.6, linewidth=0.7)
    ax.set_title("Vibration Over Time by Machine")
    ax.set_ylabel("Vibration (mm/s)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / "vibration_over_time.png", dpi=100)
    plt.close(fig)

    # 3. Production quality vs downtime relationship.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(integrated_df["downtime_minutes"], integrated_df["quality_score"], alpha=0.4, s=10)
    ax.set_xlabel("Downtime (minutes)")
    ax.set_ylabel("Quality score")
    ax.set_title("Quality Score vs Downtime")
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_downtime.png", dpi=100)
    plt.close(fig)

    # 4. Correlation heatmap of key sensor means.
    sensor_cols = [c for c in integrated_df.columns if c.endswith("_mean")]
    if sensor_cols:
        fig, ax = plt.subplots(figsize=(6, 5))
        corr = integrated_df[sensor_cols].corr(numeric_only=True)
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(sensor_cols)))
        ax.set_xticklabels(sensor_cols, rotation=90, fontsize=7)
        ax.set_yticks(range(len(sensor_cols)))
        ax.set_yticklabels(sensor_cols, fontsize=7)
        fig.colorbar(im)
        ax.set_title("Sensor Feature Correlation")
        fig.tight_layout()
        fig.savefig(out_dir / "sensor_correlation_heatmap.png", dpi=100)
        plt.close(fig)

    return sorted(p.name for p in out_dir.glob("*.png"))
