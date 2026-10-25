"""
CICIDS2017 future-attack forecasting — practical binary baseline.

Goal:
  Use the previous 10 consecutive one-minute traffic windows to predict
  whether the NEXT minute contains an attack (ATTACK vs BENIGN).

Why binary first:
  The existing 15-class temporal test split has inadequate attack-class coverage.
  A binary forecast is an actionable first milestone and allows meaningful
  attack recall measurement. It does NOT pretend to predict attack subtype.

Data handling:
  - Uses the eight CICIDS2017 capture files under DATA/raw/CICID.
  - Builds one-minute aggregates and 10-minute histories per capture.
  - Splits each capture chronologically before creating windows.
  - Balances TRAINING data only by random oversampling; validation/test stay real.
  - Trains XGBoost and RandomForest; selects a decision threshold on validation.
  - Saves metrics, predictions, fitted model, and preprocessing config.

Run from project root:
  .\\.venv\\Scripts\\python.exe -u model\\src\\train_cicids_binary_forecast.py
"""

from pathlib import Path
from collections import Counter
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "DATA" / "raw" / "CICID"
RESULT_DIR = ROOT / "model" / "results" / "cicids_binary_forecast"
ARTIFACT_DIR = ROOT / "model" / "artifacts" / "cicids_binary_forecast"

CAPTURES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]
HISTORY_MINUTES = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEED = 42


def parse_timestamps(series):
    raw = series.astype("string").str.strip()
    try:
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, errors="coerce")
    missing = parsed.isna() & raw.notna() & raw.ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(raw[missing], errors="coerce", dayfirst=False)
    missing = parsed.isna() & raw.notna() & raw.ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            raw[missing], errors="coerce", format="%m/%d/%Y %H:%M:%S"
        )
    return parsed


def load_capture(path):
    print(f"[READ] {path.name}", flush=True)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", encoding_errors="replace")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    time_col = next((c for c in df.columns if c.strip().lower() == "timestamp"), None)
    label_col = next((c for c in df.columns if c.strip().lower() == "label"), None)
    if time_col is None or label_col is None:
        raise ValueError(f"{path.name} must contain Timestamp and Label columns.")

    ts = parse_timestamps(df[time_col])
    labels = df[label_col].astype("string").str.strip()
    valid = ts.notna() & labels.notna() & labels.ne("")
    df = df.loc[valid].copy()
    df["_timestamp"] = ts.loc[valid].values
    df["_label"] = labels.loc[valid].astype(str).values
    df["_minute"] = pd.to_datetime(df["_timestamp"]).dt.floor("min")

    excluded_exact = {
        time_col, label_col, "Flow ID", "Source IP", "Destination IP",
        "Source_IP", "Destination_IP", "src_ip", "dst_ip",
    }
    numeric_cols = []
    for col in df.columns:
        if col.startswith("_") or col in excluded_exact:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            df[col] = numeric.replace([np.inf, -np.inf], np.nan)
            numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError(f"No numeric features found in {path.name}")

    group = df.groupby("_minute", sort=True)
    minute_features = group[numeric_cols].mean()
    minute_features["flow_count"] = group.size().astype(float)

    def is_attack(label):
        return str(label).strip().upper() not in {"BENIGN", "NORMAL", "0", "NONE", "NAN", ""}

    attack_by_minute = group["_label"].apply(lambda s: int(any(is_attack(x) for x in s)))
    minute_features["target"] = attack_by_minute.astype(np.int8)
    minute_features = minute_features.replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in minute_features.columns if c != "target"]
    minute_features[feature_cols] = minute_features[feature_cols].ffill().fillna(0.0)
    minute_features = minute_features.sort_index()
    print(
        f"[AGG] {path.name}: {len(df):,} rows -> {len(minute_features):,} minute bins; "
        f"attack minutes={int(minute_features['target'].sum()):,}",
        flush=True,
    )
    return minute_features, numeric_cols


def make_windows(minute_df, feature_cols, capture_name, split_name):
    if len(minute_df) <= HISTORY_MINUTES:
        return None
    values = minute_df[feature_cols].to_numpy(dtype=np.float32)
    labels = minute_df["target"].to_numpy(dtype=np.int8)
    times = minute_df.index
    xs, ys, meta = [], [], []
    for target_i in range(HISTORY_MINUTES, len(minute_df)):
        start_i = target_i - HISTORY_MINUTES
        expected = pd.date_range(times[start_i], periods=HISTORY_MINUTES + 1, freq="min")
        observed = times[start_i:target_i + 1]
        if len(observed) != HISTORY_MINUTES + 1 or not np.array_equal(observed.asi8, expected.asi8):
            continue
        hist = values[start_i:target_i]
        # Temporal summaries: last, mean, standard deviation, first-to-last delta.
        engineered = np.concatenate([
            hist[-1],
            hist.mean(axis=0),
            hist.std(axis=0),
            hist[-1] - hist[0],
        ])
        xs.append(engineered)
        ys.append(labels[target_i])
        meta.append({
            "capture": capture_name,
            "target_timestamp": times[target_i].isoformat(),
            "split": split_name,
        })
    if not xs:
        return None
    return np.vstack(xs), np.asarray(ys, dtype=np.int8), meta


def oversample_binary(X, y, seed=SEED):
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError(
            f"Training split contains only class {classes.tolist()}; cannot train attack forecast."
        )
    target_count = int(counts.max())
    indices = []
    for cls in classes:
        cls_idx = np.flatnonzero(y == cls)
        sampled = rng.choice(cls_idx, size=target_count, replace=(len(cls_idx) < target_count))
        indices.extend(sampled.tolist())
    indices = np.asarray(indices)
    rng.shuffle(indices)
    return X[indices], y[indices], {
        str(int(cls)): {"original": int(count), "after_oversampling": target_count}
        for cls, count in zip(classes, counts)
    }


def choose_threshold(y_true, probabilities):
    # Favor attack recall while requiring at least modest precision where possible.
    candidates = np.linspace(0.10, 0.90, 81)
    valid = []
    all_scores = []
    for threshold in candidates:
        pred = (probabilities >= threshold).astype(np.int8)
        p = precision_score(y_true, pred, zero_division=0)
        r = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        all_scores.append((float(threshold), float(p), float(r), float(f1)))
        if p >= 0.25:
            valid.append((r, f1, threshold, p))
    if valid:
        # Highest attack recall first; F1 breaks ties.
        best = max(valid, key=lambda item: (item[0], item[1]))
        return float(best[2]), all_scores
    # If the validation data cannot reach 25% precision, select maximum F1.
    best = max(all_scores, key=lambda item: (item[3], item[2]))
    return best[0], all_scores


def evaluate(y_true, probs, threshold):
    pred = (probs >= threshold).astype(np.int8)
    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "attack_precision": float(precision_score(y_true, pred, pos_label=1, zero_division=0)),
        "attack_recall": float(recall_score(y_true, pred, pos_label=1, zero_division=0)),
        "attack_f1": float(f1_score(y_true, pred, pos_label=1, zero_division=0)),
        "macro_f1_binary": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "confusion_matrix_labels_0_benign_1_attack": confusion_matrix(
            y_true, pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true, pred, labels=[0, 1], target_names=["BENIGN", "ATTACK"],
            output_dict=True, zero_division=0,
        ),
        "true_benign": int((y_true == 0).sum()),
        "true_attack": int((y_true == 1).sum()),
        "predicted_benign": int((pred == 0).sum()),
        "predicted_attack": int((pred == 1).sum()),
    }
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probs))
        result["average_precision"] = float(average_precision_score(y_true, probs))
    else:
        result["roc_auc"] = None
        result["average_precision"] = None
    return result, pred


def main():
    start = time.time()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    missing = [name for name in CAPTURES if not (SOURCE_DIR / name).exists()]
    if len(missing) == len(CAPTURES):
        raise FileNotFoundError(
            f"No CICIDS capture CSVs found under {SOURCE_DIR}. Missing examples: {missing[:2]}"
        )
    if missing:
        print(f"[WARN] Missing captures skipped: {missing}", flush=True)

    raw_captures = []
    feature_union = []
    capture_reports = []
    for name in CAPTURES:
        path = SOURCE_DIR / name
        if not path.exists():
            continue
        minute_df, numeric_cols = load_capture(path)
        raw_captures.append((name, minute_df))
        for col in numeric_cols + ["flow_count"]:
            if col not in feature_union:
                feature_union.append(col)

    # Make one stable feature schema, then split each capture chronologically.
    split_rows = {"train": [], "validation": [], "test": []}
    metadata = []
    for name, minute_df in raw_captures:
        minute_df = minute_df.reindex(columns=feature_union + ["target"])
        minute_df[feature_union] = minute_df[feature_union].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        n = len(minute_df)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
        partitions = {
            "train": minute_df.iloc[:train_end],
            "validation": minute_df.iloc[train_end:val_end],
            "test": minute_df.iloc[val_end:],
        }
        report = {"capture": name, "minute_bins": n, "splits": {}}
        for split_name, part in partitions.items():
            made = make_windows(part, feature_union, name, split_name)
            if made is None:
                report["splits"][split_name] = {"windows": 0, "benign": 0, "attack": 0}
                continue
            X, y, metas = made
            split_rows[split_name].append((X, y))
            metadata.extend([{**m, "true_target": int(target)} for m, target in zip(metas, y)])
            report["splits"][split_name] = {
                "windows": int(len(y)), "benign": int((y == 0).sum()), "attack": int((y == 1).sum())
            }
        capture_reports.append(report)

    def combine(name):
        chunks = split_rows[name]
        if not chunks:
            raise ValueError(f"No {name} forecasting windows were created. Check CICIDS files.")
        return np.concatenate([x for x, _ in chunks]), np.concatenate([y for _, y in chunks])

    X_train, y_train = combine("train")
    X_val, y_val = combine("validation")
    X_test, y_test = combine("test")

    for split_name, y in [("train", y_train), ("validation", y_val), ("test", y_test)]:
        print(
            f"[{split_name.upper()}] windows={len(y):,}, benign={(y == 0).sum():,}, "
            f"attack={(y == 1).sum():,}",
            flush=True,
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training windows contain only one class; cannot train a binary model.")
    if len(np.unique(y_val)) < 2:
        print("[WARN] Validation has one class; threshold selection will use available metrics.", flush=True)

    # Oversample only training rows. Never rebalance validation or test.
    X_bal, y_bal, balance_report = oversample_binary(X_train, y_train)

    models = {
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.035,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=2,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=SEED,
            n_jobs=4,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }

    results = {}
    fitted = {}
    for name, model in models.items():
        print(f"[TRAIN] {name} on {len(y_bal):,} balanced training examples...", flush=True)
        model.fit(X_bal, y_bal)
        val_probs = model.predict_proba(X_val)[:, 1]
        threshold, threshold_scores = choose_threshold(y_val, val_probs)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_metrics, test_pred = evaluate(y_test, test_probs, threshold)
        val_metrics, _ = evaluate(y_val, val_probs, threshold)
        results[name] = {
            "selected_threshold_from_validation": threshold,
            "validation_metrics_at_selected_threshold": val_metrics,
            "test_metrics": test_metrics,
            "validation_threshold_scan": threshold_scores,
        }
        fitted[name] = model
        print(
            f"[RESULT] {name}: test attack recall={test_metrics['attack_recall']:.4f}, "
            f"precision={test_metrics['attack_precision']:.4f}, "
            f"F1={test_metrics['attack_f1']:.4f}, "
            f"accuracy={test_metrics['accuracy']:.4f}",
            flush=True,
        )

    # Select the model using validation attack F1; do not select by test performance.
    chosen_name = max(
        results,
        key=lambda name: results[name]["validation_metrics_at_selected_threshold"]["attack_f1"],
    )
    chosen_model = fitted[chosen_name]
    chosen_threshold = results[chosen_name]["selected_threshold_from_validation"]
    joblib.dump(
        {"model": chosen_model, "threshold": chosen_threshold, "feature_columns": feature_union,
         "history_minutes": HISTORY_MINUTES, "task": "next-minute binary attack forecast"},
        ARTIFACT_DIR / "best_binary_forecaster.joblib",
    )

    chosen_probs = chosen_model.predict_proba(X_test)[:, 1]
    chosen_metrics, chosen_pred = evaluate(y_test, chosen_probs, chosen_threshold)
    # Attach metadata aligned by split.
    test_meta = [m for m in metadata if m["split"] == "test"]
    if len(test_meta) != len(y_test):
        test_meta = [{"capture": "", "target_timestamp": "", "split": "test"} for _ in y_test]
    pd.DataFrame({
        "capture": [m["capture"] for m in test_meta],
        "target_timestamp": [m["target_timestamp"] for m in test_meta],
        "true_target": y_test,
        "predicted_target": chosen_pred,
        "attack_probability": chosen_probs,
        "threshold": chosen_threshold,
    }).to_csv(RESULT_DIR / "test_predictions.csv", index=False)

    report = {
        "dataset": "CICIDS2017",
        "task": "Forecast whether the next 1-minute bin contains any attack (ATTACK vs BENIGN).",
        "history_minutes": HISTORY_MINUTES,
        "forecast_horizon_minutes": 1,
        "feature_aggregation": "per-minute mean numeric flow features + flow_count; window summaries last/mean/std/delta",
        "split_policy": "Each capture split chronologically 70/15/15 before window creation; no windows cross split boundaries.",
        "training_balance": {
            "method": "random oversampling of training rows only",
            "before": {str(int(c)): int(n) for c, n in zip(*np.unique(y_train, return_counts=True))},
            "after": balance_report,
            "validation_and_test_resampled": False,
        },
        "capture_split_counts": capture_reports,
        "feature_count_per_minute": len(feature_union),
        "engineered_feature_count": int(X_train.shape[1]),
        "training_windows": int(len(y_train)),
        "balanced_training_windows": int(len(y_bal)),
        "validation_windows": int(len(y_val)),
        "test_windows": int(len(y_test)),
        "models": results,
        "selected_model_by_validation_attack_f1": chosen_name,
        "selected_model_test_metrics": chosen_metrics,
        "limitations": [
            "This is a binary attack/no-attack forecaster, not an attack-subtype classifier.",
            "Oversampling is applied to training only; validation/test preserve observed prevalence.",
            "Synthetic or oversampled training examples do not guarantee high real-world recall.",
            "Results apply to CICIDS2017 captures and need external validation before deployment.",
        ],
        "runtime_seconds": round(time.time() - start, 2),
    }
    (RESULT_DIR / "forecast_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULT_DIR / "forecast_report.txt").write_text(
        "\n".join([
            "CICIDS2017 NEXT-MINUTE ATTACK FORECAST",
            f"Selected model: {chosen_name}",
            f"Test attack recall: {chosen_metrics['attack_recall']:.4f}",
            f"Test attack precision: {chosen_metrics['attack_precision']:.4f}",
            f"Test attack F1: {chosen_metrics['attack_f1']:.4f}",
            f"Test accuracy: {chosen_metrics['accuracy']:.4f}",
            f"Test ROC-AUC: {chosen_metrics['roc_auc']}",
            f"Test average precision: {chosen_metrics['average_precision']}",
            f"Test windows: {len(y_test)}",
            "",
            "Per-model test metrics:",
            *[
                f"{model_name}: recall={r['test_metrics']['attack_recall']:.4f}, "
                f"precision={r['test_metrics']['attack_precision']:.4f}, "
                f"F1={r['test_metrics']['attack_f1']:.4f}, "
                f"accuracy={r['test_metrics']['accuracy']:.4f}"
                for model_name, r in results.items()
            ],
        ]),
        encoding="utf-8",
    )
    print("\n=== FINAL SELECTED FORECASTER ===")
    print(f"Model: {chosen_name}")
    print(f"Test attack recall: {chosen_metrics['attack_recall']:.4f}")
    print(f"Test attack precision: {chosen_metrics['attack_precision']:.4f}")
    print(f"Test attack F1: {chosen_metrics['attack_f1']:.4f}")
    print(f"Test accuracy: {chosen_metrics['accuracy']:.4f}")
    print(f"Report: {RESULT_DIR / 'forecast_report.txt'}")
    print(f"JSON:   {RESULT_DIR / 'forecast_report.json'}")
    print(f"Model:  {ARTIFACT_DIR / 'best_binary_forecaster.joblib'}")


if __name__ == "__main__":
    main()
