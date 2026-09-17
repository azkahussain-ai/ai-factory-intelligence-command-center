"""Run the complete Stage I pipeline: generate -> validate -> clean ->
integrate -> engineer features -> split -> EDA -> write outputs.

Usage:
    python -m src.run_stage1
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import config
from src.data.generate_synthetic import save_raw_outputs
from src.data.load_data import (
    load_raw_maintenance_data,
    load_raw_production_data,
    load_raw_sensor_data,
)
from src.preprocessing.clean_data import (
    clean_maintenance_data,
    clean_production_data,
    clean_sensor_data,
)
from src.preprocessing.integrate_data import integrate_datasets
from src.features.build_features import build_feature_set
from src.features.split_data import (
    fit_train_only_scaler,
    save_scaler_stats,
    save_splits,
    time_based_split,
)
from src.utils.eda import (
    correlation_matrix,
    dataset_overview,
    failure_rate_by_machine,
    generate_plots,
    numeric_summary,
)


def main():
    report = {"stage": "Stage I - Data Engineering & EDA"}

    print("1/7 Generating synthetic raw data...")
    save_raw_outputs()

    print("2/7 Loading + validating raw data...")
    sensor_raw = load_raw_sensor_data(config.RAW_DIR)
    production_raw = load_raw_production_data(config.RAW_DIR)
    maintenance_raw = load_raw_maintenance_data(config.RAW_DIR)

    print("3/7 Cleaning...")
    sensor_clean, sensor_clean_summary = clean_sensor_data(sensor_raw)
    production_clean, production_clean_summary = clean_production_data(production_raw)
    maintenance_clean, maintenance_clean_summary = clean_maintenance_data(maintenance_raw)
    report["cleaning"] = {
        "sensor": sensor_clean_summary,
        "production": production_clean_summary,
        "maintenance": maintenance_clean_summary,
    }

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    sensor_clean.to_csv(config.PROCESSED_DIR / "cleaned_sensor_data.csv", index=False)
    production_clean.to_csv(config.PROCESSED_DIR / "cleaned_production.csv", index=False)
    maintenance_clean.to_csv(config.PROCESSED_DIR / "cleaned_maintenance.csv", index=False)

    print("4/7 Integrating...")
    integrated_df, integration_summary = integrate_datasets(
        production_clean, sensor_clean, maintenance_clean)
    report["integration"] = integration_summary
    integrated_df.to_csv(config.PROCESSED_DIR / "integrated_dataset.csv", index=False)

    print("5/7 Feature engineering...")
    feature_df, feature_summary = build_feature_set(integrated_df)
    report["feature_engineering"] = feature_summary

    print("6/7 Splitting (time-based)...")
    train, val, test, split_summary = time_based_split(feature_df)
    report["split"] = split_summary
    save_splits(train, val, test, config.PROCESSED_DIR)

    numeric_feature_cols = [
        c for c in feature_df.columns
        if feature_df[c].dtype.kind in "fi" and c not in ("failure_next_shift", "sim_day")
    ]
    scaler_stats = fit_train_only_scaler(train, numeric_feature_cols)
    save_scaler_stats(scaler_stats, config.PROCESSED_DIR / "train_scaler_stats.json")
    report["scaler"] = {"fitted_on": "train only", "n_features_scaled": len(scaler_stats)}

    print("7/7 EDA...")
    key_numeric_cols = [
        "production_rate", "production_count", "quality_score", "downtime_minutes",
        "vibration_mm_s_mean", "temperature_k_mean", "pressure_bar_mean",
    ]
    eda_report = {
        "sensor_overview": dataset_overview(sensor_clean),
        "production_overview": dataset_overview(production_clean),
        "maintenance_overview": dataset_overview(maintenance_clean),
        "integrated_overview": dataset_overview(integrated_df),
        "numeric_summary": numeric_summary(integrated_df, key_numeric_cols),
        "failure_rate_by_machine": failure_rate_by_machine(feature_df, "failure_next_shift"),
        "sensor_correlation": correlation_matrix(
            integrated_df, [c for c in integrated_df.columns if c.endswith("_mean")]),
    }
    plot_dir = config.REPORTS_DIR / "figures"
    eda_report["plots_generated"] = generate_plots(integrated_df, sensor_clean, plot_dir)
    report["eda"] = eda_report

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "stage1_eda_summary.json").write_text(json.dumps(report, indent=2, default=str))

    print("\nStage I complete.")
    print(f"  sensor rows (clean): {len(sensor_clean)}")
    print(f"  production rows (clean): {len(production_clean)}")
    print(f"  maintenance rows (clean): {len(maintenance_clean)}")
    print(f"  integrated rows: {len(integrated_df)}")
    print(f"  feature rows (with target): {len(feature_df)}")
    print(f"  train/val/test: {len(train)}/{len(val)}/{len(test)}")
    print(f"  report: {config.REPORTS_DIR / 'stage1_eda_summary.json'}")
    return report


if __name__ == "__main__":
    main()
