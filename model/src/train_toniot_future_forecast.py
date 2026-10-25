from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "DATA/raw/TON_IoT/Network/train_test_network.csv"
OUT = ROOT / "model/results/toniot_future_forecasting"
ART = ROOT / "model/artifacts/toniot_future_forecasting"

HISTORY = 5
FREQ = "1min"
SEED = 42
TIMESTAMP_CANDIDATES = ("ts", "timestamp", "time", "datetime", "date")


def parse_ts(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= 0.8:
        unit = "ms" if numeric.dropna().abs().median() > 1e12 else "s"
        return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)
    return pd.to_datetime(series, errors="coerce", utc=True)


def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA}")

    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print(f"Loading {DATA}", flush=True)
    df = pd.read_csv(DATA, low_memory=False)

    target_col = "type" if "type" in df.columns else None
    ts_col = next((col for col in TIMESTAMP_CANDIDATES if col in df.columns), None)

    if target_col is None:
        raise ValueError(
            f"Missing target column 'type'. Found columns: {df.columns.tolist()}"
        )

    if ts_col is None:
        raise ValueError(
            "No timestamp column found in this TON_IoT network CSV. "
            "A chronological future-forecasting split cannot be established "
            "from this file alone. Use a timestamped source or a documented "
            "ordering that preserves time. The script will not shuffle rows "
            "or invent timestamps."
        )

    ts = parse_ts(df[ts_col])
    valid = ts.notna() & df[target_col].notna()
    dropped = int((~valid).sum())
    df = df.loc[valid].copy()
    df["_time"] = ts.loc[valid]
    df["_minute"] = df["_time"].dt.floor(FREQ)

    raw_labels = df[target_col].astype(str).str.strip().str.lower()
    df["_label"] = raw_labels.replace(
        {"benign": "normal", "benign traffic": "normal", "0": "normal"}
    )

    excluded = {
        ts_col, target_col, "label", "src_ip", "dst_ip", "src_mac", "dst_mac",
        "_time", "_minute", "_label",
    }
    numeric = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in excluded and not col.startswith("Unnamed:")
    ]
    if not numeric:
        raise ValueError("No numeric traffic features found.")

    grouped = df.groupby("_minute", sort=True)
    features = grouped[numeric].agg(["mean", "std", "max", "sum"])
    features.columns = [f"{col}__{stat}" for col, stat in features.columns]
    features["event_count"] = grouped.size().astype(float)

    def minute_target(series):
        series = series[~series.isin(["unknown", "nan", "none", ""])]
        attacks = series[series != "normal"]
        return attacks.value_counts().index[0] if len(attacks) else "normal"

    labels = grouped["_label"].apply(minute_target).rename("target")
    table = features.join(labels).sort_index().replace([np.inf, -np.inf], np.nan)
    feature_columns = [col for col in table.columns if col != "target"]
    table[feature_columns] = table[feature_columns].fillna(0.0)

    if len(table) < 100:
        raise ValueError(
            f"Only {len(table)} minute bins; insufficient for a useful temporal split."
        )

    print(
        f"Rows={len(df):,}; minute bins={len(table):,}; "
        f"numeric inputs={len(numeric)}; dropped invalid rows={dropped}",
        flush=True,
    )

    n = len(table)
    bounds = {
        "train": (0, int(0.70 * n)),
        "validation": (int(0.70 * n), int(0.85 * n)),
        "test": (int(0.85 * n), n),
    }
    times = table.index
    minute_features = table[feature_columns].to_numpy(dtype=np.float32)
    minute_labels = table["target"].astype(str).to_numpy()

    def build_split(lo, hi):
        xs, ys, target_times, history_times = [], [], [], []
        for j in range(max(lo + HISTORY, HISTORY), hi):
            i = j - HISTORY
            observed_times = times[i:j + 1]
            expected_times = pd.date_range(
                observed_times[0], periods=HISTORY + 1, freq=FREQ
            )
            if (
                len(observed_times) != HISTORY + 1
                or not np.array_equal(observed_times.asi8, expected_times.asi8)
            ):
                continue

            xs.append(minute_features[i:j].reshape(-1))
            ys.append(minute_labels[j])
            target_times.append(times[j].isoformat())
            history_times.append([t.isoformat() for t in times[i:j]])

        X = np.asarray(xs, dtype=np.float32).reshape(
            -1, HISTORY * len(feature_columns)
        )
        return X, np.asarray(ys, dtype=str), target_times, history_times

    splits = {name: build_split(*span) for name, span in bounds.items()}
    for name, (_, y, _, _) in splits.items():
        print(f"{name}: {len(y):,} forecasting examples", flush=True)

    X_train, y_train, _, _ = splits["train"]
    X_val, y_val, _, _ = splits["validation"]
    X_test, y_test, test_times, history_times = splits["test"]

    if min(len(y_train), len(y_val), len(y_test)) < 10:
        raise ValueError(
            "Too few strictly consecutive 1-minute samples in at least one split. "
            "The dataset has timestamp gaps; no random-split fallback will be used."
        )

    encoder = LabelEncoder().fit(y_train)
    unseen_val = sorted(set(y_val) - set(encoder.classes_))
    unseen_test = sorted(set(y_test) - set(encoder.classes_))
    if unseen_val or unseen_test:
        raise ValueError(
            "Chronological split contains classes absent from training. "
            f"Validation={unseen_val}; test={unseen_test}. "
            "A closed-set model cannot learn these classes. Consider a coarser "
            "attack taxonomy or binary attack-vs-normal forecasting."
        )

    y_train_id = encoder.transform(y_train)
    y_val_id = encoder.transform(y_val)
    y_test_id = encoder.transform(y_test)
    classes = encoder.classes_.tolist()

    counts = np.bincount(y_train_id, minlength=len(classes))
    sample_weights = (len(y_train_id) / (len(classes) * np.maximum(counts, 1)))[
        y_train_id
    ]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=450,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=2.0,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
    )

    print("Training class-weighted XGBoost...", flush=True)
    train_started = time.time()
    model.fit(
        X_train,
        y_train_id,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val_id)],
        verbose=False,
    )
    train_seconds = round(time.time() - train_started, 2)

    probabilities = model.predict_proba(X_test)
    predicted_ids = probabilities.argmax(axis=1)
    predicted_labels = encoder.inverse_transform(predicted_ids)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_id,
        predicted_ids,
        labels=np.arange(len(classes)),
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test_id, predicted_ids, average="macro", zero_division=0
    )
    weighted_f1 = precision_recall_fscore_support(
        y_test_id, predicted_ids, average="weighted", zero_division=0
    )[2]

    # Reference-only persistence baseline: predict the next label from the
    # previous minute's true label. This is not a deployable model score.
    persistence_true, persistence_pred = [], []
    lo, hi = bounds["test"]
    for j in range(max(lo + HISTORY, HISTORY), hi):
        i = j - HISTORY
        observed_times = times[i:j + 1]
        expected_times = pd.date_range(
            observed_times[0], periods=HISTORY + 1, freq=FREQ
        )
        if (
            len(observed_times) == HISTORY + 1
            and np.array_equal(observed_times.asi8, expected_times.asi8)
            and minute_labels[j - 1] in encoder.classes_
        ):
            persistence_true.append(minute_labels[j])
            persistence_pred.append(minute_labels[j - 1])

    persistence = None
    if persistence_true and set(persistence_true).issubset(set(classes)):
        true_ids = encoder.transform(persistence_true)
        pred_ids = encoder.transform(persistence_pred)
        persistence = {
            "accuracy": float(accuracy_score(true_ids, pred_ids)),
            "macro_recall": float(
                precision_recall_fscore_support(
                    true_ids,
                    pred_ids,
                    labels=np.arange(len(classes)),
                    average="macro",
                    zero_division=0,
                )[1]
            ),
            "macro_f1": float(
                precision_recall_fscore_support(
                    true_ids,
                    pred_ids,
                    labels=np.arange(len(classes)),
                    average="macro",
                    zero_division=0,
                )[2]
            ),
            "note": (
                "Oracle persistence using the previous ground-truth label; "
                "reference only, not a deployable score."
            ),
        }

    report = {
        "task": (
            "Predict attack type in the next 1-minute bin using the previous "
            "5 consecutive 1-minute bins."
        ),
        "dataset": str(DATA.relative_to(ROOT)),
        "timestamp_column": ts_col,
        "target_column": target_col,
        "raw_rows_after_timestamp_filter": int(len(df)),
        "dropped_invalid_rows": dropped,
        "minute_bins": int(len(table)),
        "history_minutes": HISTORY,
        "frequency": FREQ,
        "numeric_input_columns": numeric,
        "aggregated_features_per_minute": int(len(feature_columns)),
        "flattened_features": int(X_train.shape[1]),
        "classes": classes,
        "split_boundaries": {
            "train_end": times[bounds["train"][1] - 1].isoformat(),
            "validation_end": times[bounds["validation"][1] - 1].isoformat(),
            "test_start": times[bounds["test"][0]].isoformat(),
            "test_end": times[-1].isoformat(),
        },
        "samples": {name: int(len(split[1])) for name, split in splits.items()},
        "train_class_counts": {
            str(label): int(count) for label, count in zip(classes, counts)
        },
        "test_class_counts": {
            str(label): int(count) for label, count in pd.Series(y_test).value_counts().items()
        },
        "unseen_validation_classes": unseen_val,
        "unseen_test_classes": unseen_test,
        "xgboost": {
            "accuracy": float(accuracy_score(y_test_id, predicted_ids)),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "train_seconds": train_seconds,
            "per_class": {
                label: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i]),
                }
                for i, label in enumerate(classes)
            },
            "confusion_matrix": confusion_matrix(
                y_test_id, predicted_ids, labels=np.arange(len(classes))
            ).tolist(),
            "classification_report": classification_report(
                y_test_id,
                predicted_ids,
                labels=np.arange(len(classes)),
                target_names=classes,
                output_dict=True,
                zero_division=0,
            ),
        },
        "persistence_baseline": persistence,
        "runtime_seconds": round(time.time() - started, 2),
        "limitations": [
            "Uses numeric traffic features only in this first baseline.",
            "Minute label is the most common non-normal type if an attack occurs, otherwise normal.",
            "Only strictly consecutive observed one-minute bins are used; sequences do not cross timestamp gaps.",
            "Persistence comparison uses prior ground-truth labels and is an oracle reference, not a deployable score.",
            "A class appearing only after the chronological training period cannot be learned by this closed-set model.",
        ],
    }

    model.save_model(str(ART / "xgboost_future_attack.json"))
    (OUT / "forecast_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (OUT / "forecast_config.json").write_text(
        json.dumps(
            {
                "history_minutes": HISTORY,
                "frequency": FREQ,
                "timestamp_column": ts_col,
                "target_column": target_col,
                "numeric_input_columns": numeric,
                "aggregated_feature_columns": feature_columns,
                "classes": classes,
                "split_boundaries": report["split_boundaries"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(OUT / "per_class_report.csv", index=False)
    pd.DataFrame(
        {
            "target_time": test_times,
            "history_times": [json.dumps(values) for values in history_times],
            "true_attack_type": y_test,
            "predicted_attack_type": predicted_labels,
            "confidence": probabilities.max(axis=1),
        }
    ).to_csv(OUT / "test_predictions.csv", index=False)

    print("\n=== TON_IoT FUTURE-ATTACK BASELINE ===")
    print(f"Accuracy: {accuracy_score(y_test_id, predicted_ids):.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall: {macro_recall:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Training seconds: {train_seconds}")
    print("\nPer-class results:")
    print(pd.read_csv(OUT / "per_class_report.csv").to_string(index=False))
    print("\nOracle persistence reference:", json.dumps(persistence, indent=2))
    print(f"\nReport: {OUT / 'forecast_report.json'}")
    print(f"Model:  {ART / 'xgboost_future_attack.json'}")


if __name__ == "__main__":
    main()
