"""
Compare CICIDS2017 binary next-minute forecasting:
1) XGBoost with random oversampling (baseline)
2) XGBoost with class weighting (no duplicated training rows)

Uses the same data-building approach as train_cicids_binary_forecast.py:
- each CICIDS capture is aggregated to one-minute bins
- 10 consecutive minutes predict whether the next minute contains an attack
- chronological split per capture (70/15/15) before making windows
- validation/test are never resampled
- threshold is selected on validation only
- test set is used only for final reporting

Run from repository root:
    .\.venv\Scripts\python.exe -u model\src\compare_cicids_imbalance.py
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.utils import resample
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR_CANDIDATES = [ROOT / "DATA" / "raw" / "CICID", ROOT / "DATA" / "raw" / "CICIDS"]
OUT_DIR = ROOT / "model" / "results" / "cicids_imbalance_comparison"
HISTORY_MINUTES = 10
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

CAPTURE_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


def locate_raw_dir():
    for candidate in RAW_DIR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find CICIDS raw data. Expected DATA/raw/CICID (or DATA/raw/CICIDS)."
    )


def normalize_columns(df):
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    return df


def read_capture(path):
    # CICIDS CSV files often contain whitespace in column names and labels.
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        # CICIDS2017 exports may contain Windows-1252 bytes in otherwise numeric CSVs.
        df = pd.read_csv(path, low_memory=False, encoding="cp1252")
    df = normalize_columns(df)

    time_col = next((c for c in df.columns if c.lower() in ("timestamp", "time")), None)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if time_col is None or label_col is None:
        raise ValueError(f"{path.name}: could not find Timestamp and Label columns.")

    timestamps = pd.to_datetime(df[time_col].astype(str).str.strip(), errors="coerce", dayfirst=False)
    # Some CICIDS exports use day/month/year; retry those rows that failed.
    bad = timestamps.isna()
    if bad.any():
        timestamps.loc[bad] = pd.to_datetime(
            df.loc[bad, time_col].astype(str).str.strip(),
            errors="coerce", dayfirst=True
        )
    valid = timestamps.notna()
    df = df.loc[valid].copy()
    timestamps = timestamps.loc[valid]

    labels = df[label_col].astype(str).str.strip().str.upper()
    is_benign = labels.isin(["BENIGN", "NORMAL", "0"])
    df["_target"] = (~is_benign).astype(np.int8)
    df["_minute"] = timestamps.dt.floor("min").to_numpy()

    # Numeric network-flow features only. Exclude target/time and known identifier-like fields.
    exclude = {time_col, label_col, "_target", "_minute"}
    numeric_cols = []
    for col in df.columns:
        if col in exclude or col.startswith("_"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        # Retain columns with at least some real numeric values.
        if converted.notna().any():
            df[col] = converted.replace([np.inf, -np.inf], np.nan)
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError(f"{path.name}: no numeric feature columns found.")

    # Drop fields that are effectively IDs or addresses when represented as numeric.
    filtered = []
    for col in numeric_cols:
        low = col.lower()
        if low in {"flow id", "source ip", "destination ip", "src ip", "dst ip"}:
            continue
        filtered.append(col)
    numeric_cols = filtered

    df["_target"] = df["_target"].astype(np.int8)
    agg = df.groupby("_minute", sort=True).agg(
        {**{c: "mean" for c in numeric_cols}, "_target": "max"}
    )
    counts = df.groupby("_minute", sort=True).size().rename("flow_count")
    agg = agg.join(counts)
    agg = agg.replace([np.inf, -np.inf], np.nan)
    agg["flow_count"] = agg["flow_count"].astype(float)
    return agg, numeric_cols


def make_windows(minute_df, feature_cols):
    """Make 10-minute history -> next-minute binary target, only across consecutive minutes."""
    if len(minute_df) <= HISTORY_MINUTES:
        return np.empty((0, len(feature_cols) * 4), dtype=np.float32), np.empty(0, dtype=np.int8)

    x_rows, y_rows = [], []
    times = minute_df.index
    vals = minute_df[feature_cols].to_numpy(dtype=np.float32)
    targets = minute_df["target"].to_numpy(dtype=np.int8)

    for end in range(HISTORY_MINUTES, len(minute_df)):
        # Require all 10 history minutes and the target minute to be consecutive.
        segment = times[end - HISTORY_MINUTES:end + 1]
        diffs = np.diff(segment.values).astype("timedelta64[m]").astype(np.int64)
        if not np.all(diffs == 1):
            continue
        hist = vals[end - HISTORY_MINUTES:end]
        # Summary representation matches the baseline's last/mean/std/delta features.
        last = hist[-1]
        mean = np.nanmean(hist, axis=0)
        std = np.nanstd(hist, axis=0)
        delta = hist[-1] - hist[0]
        feat = np.concatenate([last, mean, std, delta])
        x_rows.append(np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0))
        y_rows.append(targets[end])

    if not x_rows:
        return np.empty((0, len(feature_cols) * 4), dtype=np.float32), np.empty(0, dtype=np.int8)
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int8)


def build_dataset():
    raw_dir = locate_raw_dir()
    existing = [raw_dir / f for f in CAPTURE_FILES if (raw_dir / f).exists()]
    if not existing:
        raise FileNotFoundError(f"No expected CICIDS capture CSVs found in {raw_dir}")

    capture_parts = []
    all_features = set()
    for path in existing:
        print(f"[READ] {path.name}", flush=True)
        agg, features = read_capture(path)
        agg["capture"] = path.name
        capture_parts.append(agg)
        all_features.update(features)
        attacks = int(agg["_target"].sum())
        print(f"[AGG] {path.name}: {len(agg):,} minute bins; attack minutes={attacks}", flush=True)

    feature_cols = sorted(all_features) + ["flow_count"]
    split_sets = {k: {"X": [], "y": []} for k in ("train", "val", "test")}

    for agg in capture_parts:
        # Fill absent capture-specific feature columns consistently.
        agg = agg.reindex(columns=sorted(all_features) + ["_target", "flow_count"])
        agg = agg.replace([np.inf, -np.inf], np.nan)
        # Do not let missing feature values leak labels; missing values become zero.
        agg[sorted(all_features) + ["flow_count"]] = agg[sorted(all_features) + ["flow_count"]].fillna(0)
        agg = agg.rename(columns={"_target": "target"})

        n = len(agg)
        train_end = int(n * TRAIN_FRACTION)
        val_end = int(n * (TRAIN_FRACTION + VAL_FRACTION))
        chronological = {
            "train": agg.iloc[:train_end].copy(),
            "val": agg.iloc[train_end:val_end].copy(),
            "test": agg.iloc[val_end:].copy(),
        }

        for split_name, part in chronological.items():
            X, y = make_windows(part, feature_cols)
            if len(y):
                split_sets[split_name]["X"].append(X)
                split_sets[split_name]["y"].append(y)

    result = {}
    for name in ("train", "val", "test"):
        xs = split_sets[name]["X"]
        ys = split_sets[name]["y"]
        result[name] = (
            np.concatenate(xs, axis=0) if xs else np.empty((0, len(feature_cols) * 4), dtype=np.float32),
            np.concatenate(ys, axis=0) if ys else np.empty(0, dtype=np.int8),
        )
        X, y = result[name]
        print(f"[{name.upper()}] windows={len(y):,}, benign={int((y==0).sum()):,}, attack={int((y==1).sum()):,}", flush=True)

    if any(len(result[k][1]) == 0 for k in result):
        raise RuntimeError("At least one split has no windows; cannot compare models.")
    return result


def oversample_training(X, y):
    df = pd.DataFrame(X)
    df["_target"] = y
    majority = df[df["_target"] == 0]
    minority = df[df["_target"] == 1]
    if len(minority) == 0 or len(majority) == 0:
        raise RuntimeError("Training split must contain both benign and attack windows.")
    minority_up = resample(
        minority, replace=True, n_samples=len(majority),
        random_state=RANDOM_STATE
    )
    balanced = pd.concat([majority, minority_up], axis=0).sample(
        frac=1.0, random_state=RANDOM_STATE
    )
    return balanced.drop(columns="_target").to_numpy(dtype=np.float32), balanced["_target"].to_numpy(dtype=np.int8)


def make_model(scale_pos_weight=1.0):
    return XGBClassifier(
        n_estimators=450,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=float(scale_pos_weight),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


TARGET_VALIDATION_RECALL = 0.95

def choose_threshold(y_true, probs):
    # One bounded final attempt: among thresholds with validation recall >= 95%,
    # choose the one with the highest precision. Validation only; never inspect test
    # outcomes when choosing this threshold. If no threshold reaches the target,
    # choose the threshold with the highest recall, then precision.
    eligible = []
    all_candidates = []
    for threshold in np.arange(0.01, 0.951, 0.01):
        pred = (probs >= threshold).astype(np.int8)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f = f1_score(y_true, pred, zero_division=0)
        candidate = {
            "threshold": float(threshold),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
        all_candidates.append(candidate)
        if r >= TARGET_VALIDATION_RECALL:
            eligible.append(candidate)

    if eligible:
        best = max(eligible, key=lambda c: (c["precision"], c["recall"], c["f1"], c["threshold"]))
        rule = "highest validation precision among thresholds with validation recall >= 0.95"
    else:
        best = max(all_candidates, key=lambda c: (c["recall"], c["precision"], c["f1"], c["threshold"]))
        rule = "no threshold met 0.95 validation recall; highest validation recall then precision"

    return best["threshold"], {
        "validation_precision": best["precision"],
        "validation_recall": best["recall"],
        "validation_f1": best["f1"],
        "threshold_selection_rule": rule,
        "validation_recall_target": TARGET_VALIDATION_RECALL,
    }


def score_model(name, model, X_val, y_val, X_test, y_test):
    val_probs = model.predict_proba(X_val)[:, 1]
    threshold, val_metrics = choose_threshold(y_val, val_probs)
    test_probs = model.predict_proba(X_test)[:, 1]
    pred = (test_probs >= threshold).astype(np.int8)

    metrics = {
        "model": name,
        "threshold_selected_on_validation": threshold,
        **val_metrics,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_attack_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_attack_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_attack_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) == 2 else None,
        "test_pr_auc_average_precision": float(average_precision_score(y_test, test_probs)) if len(np.unique(y_test)) == 2 else None,
        "test_confusion_matrix_labels_0_1": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "test_classification_report": classification_report(
            y_test, pred, labels=[0, 1], target_names=["BENIGN", "ATTACK"],
            output_dict=True, zero_division=0
        ),
    }
    print(
        f"[RESULT] {name}: threshold={threshold:.2f}, "
        f"attack P={metrics['test_attack_precision']:.4f}, "
        f"R={metrics['test_attack_recall']:.4f}, "
        f"F1={metrics['test_attack_f1']:.4f}, "
        f"PR-AUC={metrics['test_pr_auc_average_precision']:.4f}, "
        f"accuracy={metrics['test_accuracy']:.4f}",
        flush=True,
    )
    print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics["test_confusion_matrix_labels_0_1"], flush=True)
    return metrics, model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    X_train, y_train = dataset["train"]
    X_val, y_val = dataset["val"]
    X_test, y_test = dataset["test"]

    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"[INFO] Original training distribution: benign={n_neg}, attack={n_pos}; scale_pos_weight={scale_pos_weight:.4f}", flush=True)

    # Experiment A: the current approach, random oversampling.
    X_over, y_over = oversample_training(X_train, y_train)
    print(f"[TRAIN] xgboost_oversampling: {len(y_over):,} rows; benign={int((y_over==0).sum())}, attack={int((y_over==1).sum())}", flush=True)
    model_over = make_model(scale_pos_weight=1.0)
    model_over.fit(X_over, y_over, verbose=False)
    result_over, _ = score_model("xgboost_random_oversampling", model_over, X_val, y_val, X_test, y_test)

    # Experiment B: original rows with a positive-class weight; no resampling.
    print(f"[TRAIN] xgboost_class_weight: {len(y_train):,} original rows; no oversampling", flush=True)
    model_weight = make_model(scale_pos_weight=scale_pos_weight)
    model_weight.fit(X_train, y_train, verbose=False)
    result_weight, _ = score_model("xgboost_class_weight", model_weight, X_val, y_val, X_test, y_test)

    # Select recall-first using validation only; precision breaks ties. Test metrics do not select the winner.
    candidates = [
        ("xgboost_random_oversampling", result_over),
        ("xgboost_class_weight", result_weight),
    ]
    winner_name, winner = max(
        candidates,
        key=lambda item: (item[1]["validation_recall"], item[1]["validation_precision"], item[1]["validation_f1"])
    )

    summary = {
        "dataset": "CICIDS2017",
        "task": "binary next-minute attack forecasting",
        "history_minutes": HISTORY_MINUTES,
        "split": "per-capture chronological 70/15/15; resampling on train only",
        "selection_rule": "threshold: maximize validation precision subject to validation recall >= 0.95 (fallback highest validation recall); model: validation recall then precision and F1; test not used for selection",
        "winner_by_validation": winner_name,
        "experiments": [result_over, result_weight],
        "test_set_counts": {
            "total": int(len(y_test)),
            "benign": int((y_test == 0).sum()),
            "attack": int((y_test == 1).sum()),
        },
    }
    with open(OUT_DIR / "comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "CICIDS2017 binary next-minute forecasting: imbalance comparison",
        "=" * 68,
        f"Winner selected using validation metrics: {winner_name}",
        f"History: {HISTORY_MINUTES} minutes; target: attack in the next minute",
        "Chronological split per capture: 70% train / 15% validation / 15% test",
        "Only training data was resampled; validation/test remained untouched.",
        "",
    ]
    for item in [result_over, result_weight]:
        lines.extend([
            item["model"],
            "-" * len(item["model"]),
            f"Validation threshold: {item['threshold_selected_on_validation']:.2f}",
            f"Validation precision / recall / F1: {item['validation_precision']:.4f} / {item['validation_recall']:.4f} / {item['validation_f1']:.4f}",
            f"Test attack precision / recall / F1: {item['test_attack_precision']:.4f} / {item['test_attack_recall']:.4f} / {item['test_attack_f1']:.4f}",
            f"Test macro-F1: {item['test_macro_f1']:.4f}",
            f"Test PR-AUC (average precision): {item['test_pr_auc_average_precision']:.4f}",
            f"Test ROC-AUC: {item['test_roc_auc']}",
            f"Test accuracy: {item['test_accuracy']:.4f}",
            f"Confusion matrix [[TN, FP], [FN, TP]]: {item['test_confusion_matrix_labels_0_1']}",
            "",
        ])
    lines.extend([
        "Interpretation note:",
        "Precision-focused threshold selected on validation among thresholds with recall >= 0.95; if none qualifies, highest validation recall then precision. Model winner selected on validation recall/precision/F1.",
        "The final test metrics are estimates for this chronological split and may be noisy because the test set has few attack-positive windows.",
    ])
    (OUT_DIR / "comparison_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n=== COMPARISON COMPLETE ===", flush=True)
    print(f"Winner selected using validation metrics: {winner_name}", flush=True)
    print(f"Text report: {OUT_DIR / 'comparison_report.txt'}", flush=True)
    print(f"JSON report: {OUT_DIR / 'comparison_report.json'}", flush=True)


if __name__ == "__main__":
    main()
