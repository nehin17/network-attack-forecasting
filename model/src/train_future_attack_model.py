
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "DATA" / "processed" / "future_attack_forecasting_cicids_v2"
RESULT_DIR = ROOT / "model" / "results" / "future_attack_forecasting_cicids_v2"
ARTIFACT_DIR = ROOT / "model" / "artifacts" / "future_attack_forecasting_cicids_v2"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_split(name):
    path = DATA_DIR / f"{name}_windows.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        print(f"[INFO] {name} archive keys: {list(data.files)}")
        X = data["X"]
        y = data["y"]

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(str)

    if X.ndim != 3:
        raise ValueError(f"{name}: expected 3D X, got {X.shape}")
    if len(X) != len(y):
        raise ValueError(f"{name}: X/y sample counts differ")
    if not np.isfinite(X).all():
        raise ValueError(f"{name}: X contains NaN or infinite values")

    print(
        f"[DATA] {name}: X={X.shape}, "
        f"samples={len(y)}, classes={len(np.unique(y))}"
    )
    return X, y


def evaluate(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
    }


def main():
    started = time.time()
    print("[START] Future attack forecasting model training")
    print(f"[VERSIONS] XGBoost={xgb.__version__}, sklearn={sklearn.__version__}")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # Flatten the 10-minute sequence while retaining time order.
    # Each feature vector represents 10 * 79 = 790 values.
    n_steps, n_features = X_train.shape[1], X_train.shape[2]

    if X_val.shape[1:] != X_train.shape[1:]:
        raise ValueError("Validation window shape differs from training.")
    if X_test.shape[1:] != X_train.shape[1:]:
        raise ValueError("Test window shape differs from training.")

    X_train = X_train.reshape(len(X_train), -1)
    X_val = X_val.reshape(len(X_val), -1)
    X_test = X_test.reshape(len(X_test), -1)

    # Encode labels using training labels only.
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    train_classes = list(encoder.classes_)

    print(f"[CLASSES] Training classes ({len(train_classes)}):")
    for label in train_classes:
        print(f"  - {label}: {int(np.sum(y_train == label))}")

    unseen_val = sorted(set(y_val) - set(train_classes))
    unseen_test = sorted(set(y_test) - set(train_classes))
    if unseen_val:
        print(f"[WARNING] Classes absent from train but present in val: {unseen_val}")
    if unseen_test:
        print(f"[WARNING] Classes absent from train but present in test: {unseen_test}")

    # A majority-class baseline establishes a simple reference point.
    majority_class = pd.Series(y_train).value_counts().idxmax()
    majority_val_pred = np.full(len(y_val), majority_class, dtype=str)
    majority_val_metrics = evaluate(
        y_val, majority_val_pred, sorted(set(y_val) | set(majority_val_pred))
    )

    print("\n[BASELINE] Majority-class validation metrics")
    print(json.dumps(majority_val_metrics, indent=2))
    print(f"[BASELINE] Always predicts: {majority_class}")

    # Balance the influence of training examples without altering the data.
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train_encoded,
    )

    # Choose hyperparameters on validation data only.
    candidates = []
    for depth in (2, 3, 4):
        print(f"\n[TRAIN] XGBoost max_depth={depth}")
        model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=len(train_classes),
            n_estimators=300,
            max_depth=depth,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            reg_lambda=5.0,
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=-1,
        )

        model.fit(X_train, y_train_encoded, sample_weight=sample_weights)

        val_encoded_pred = model.predict(X_val).astype(int)
        val_pred = encoder.inverse_transform(val_encoded_pred)
        val_metrics = evaluate(
            y_val,
            val_pred,
            sorted(set(y_val) | set(val_pred)),
        )

        print("[VALIDATION]", json.dumps(val_metrics, indent=2))
        candidates.append((val_metrics["macro_f1"], depth, model, val_metrics))

    # Stable tie-break: prefer the shallower model when validation scores tie.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    best_val_f1, best_depth, best_model, best_val_metrics = candidates[0]

    print(f"\n[SELECTED] max_depth={best_depth}")
    print("[SELECTED] Validation metrics:", json.dumps(best_val_metrics, indent=2))

    # Evaluate the selected model on the untouched test set exactly once.
    test_encoded_pred = best_model.predict(X_test).astype(int)
    test_pred = encoder.inverse_transform(test_encoded_pred)

    all_test_labels = sorted(set(y_test) | set(test_pred))
    test_metrics = evaluate(y_test, test_pred, all_test_labels)

    report = classification_report(
        y_test,
        test_pred,
        labels=all_test_labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test,
        test_pred,
        labels=all_test_labels,
    )

    print("\n[TEST] Final held-out test metrics")
    print(json.dumps(test_metrics, indent=2))
    print("\n[TEST] Per-class report")
    print(classification_report(
        y_test,
        test_pred,
        labels=all_test_labels,
        zero_division=0,
    ))
    print("[TEST] Confusion matrix labels:", all_test_labels)
    print(matrix)

    # Save artifacts in a separate directory; do not overwrite the current model.
    model_path = ARTIFACT_DIR / "future_attack_xgboost.joblib"
    encoder_path = ARTIFACT_DIR / "label_encoder.joblib"
    joblib.dump(best_model, model_path)
    joblib.dump(encoder, encoder_path)

    config = {
        "task": "predict attack label in the next consecutive minute",
        "dataset": "CICIDS2017",
        "window_minutes": int(n_steps),
        "features_per_minute": int(n_features),
        "flattened_input_features": int(n_steps * n_features),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "training_classes": train_classes,
        "classes_unseen_in_validation_training": unseen_val,
        "classes_unseen_in_test_training": unseen_test,
        "selected_max_depth": int(best_depth),
        "selection_metric": "validation macro F1",
        "validation_metrics": best_val_metrics,
        "majority_baseline_validation_metrics": majority_val_metrics,
        "test_metrics": test_metrics,
        "model_file": model_path.name,
        "label_encoder_file": encoder_path.name,
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
    }

    with open(ARTIFACT_DIR / "forecast_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with open(RESULT_DIR / "forecast_test_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "metrics": test_metrics,
            "classification_report": report,
            "confusion_matrix_labels": all_test_labels,
            "confusion_matrix": matrix.tolist(),
            "validation_candidates": [
                {
                    "max_depth": int(depth),
                    "validation_metrics": metrics,
                }
                for _, depth, _, metrics in candidates
            ],
        }, f, indent=2)

    predictions = pd.DataFrame({
        "actual_label": y_test,
        "predicted_label": test_pred,
        "correct": y_test == test_pred,
    })
    predictions.to_csv(RESULT_DIR / "forecast_test_predictions.csv", index=False)

    print(f"\n[SAVED] Model: {model_path}")
    print(f"[SAVED] Label encoder: {encoder_path}")
    print(f"[SAVED] Config: {ARTIFACT_DIR / 'forecast_config.json'}")
    print(f"[SAVED] Test report: {RESULT_DIR / 'forecast_test_report.json'}")
    print(f"[SAVED] Test predictions: {RESULT_DIR / 'forecast_test_predictions.csv'}")
    print(f"[DONE] Elapsed: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
