import os
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# Paths
# =========================================================

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(
    BASE, "DATA", "processed", "CICIDS"
)

RESULT_DIR = os.path.join(
    BASE, "model", "results", "xgboost"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# =========================================================
# Load frozen forecasting dataset
# =========================================================

train = np.load(
    os.path.join(DATA_DIR, "train_multiclass.npz")
)

val = np.load(
    os.path.join(DATA_DIR, "val_multiclass.npz")
)

test = np.load(
    os.path.join(DATA_DIR, "coverage_test_multiclass.npz")
)

classes = np.load(
    os.path.join(DATA_DIR, "multiclass_classes.npy"),
    allow_pickle=True
)

class_to_id = {
    name: i
    for i, name in enumerate(classes)
}


X_train_raw = train["X"].astype(np.float32)
X_val_raw = val["X"].astype(np.float32)
X_test_raw = test["X"].astype(np.float32)

y_train = np.array(
    [class_to_id[x] for x in train["y"]],
    dtype=np.int64
)

y_val = np.array(
    [class_to_id[x] for x in val["y"]],
    dtype=np.int64
)

y_test = np.array(
    [class_to_id[x] for x in test["y"]],
    dtype=np.int64
)


print("Original sequence shapes:")
print("Train:", X_train_raw.shape)
print("Val:  ", X_val_raw.shape)
print("Test: ", X_test_raw.shape)
print("Classes:", len(classes))


# =========================================================
# Temporal feature extraction
# =========================================================

def make_temporal_features(X):

    # X = [samples, 10 minutes, 70 features]

    last = X[:, -1, :]

    mean = X.mean(axis=1)

    std = X.std(axis=1)

    delta = X[:, -1, :] - X[:, 0, :]

    features = np.concatenate(
        [
            last,
            mean,
            std,
            delta
        ],
        axis=1
    )

    return features


X_train = make_temporal_features(X_train_raw)
X_val = make_temporal_features(X_val_raw)
X_test = make_temporal_features(X_test_raw)


print("\nTemporal feature shapes:")
print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)


# =========================================================
# Handle invalid values
# =========================================================

X_train = np.nan_to_num(
    X_train,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_val = np.nan_to_num(
    X_val,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test = np.nan_to_num(
    X_test,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


# =========================================================
# XGBoost
# =========================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(classes),

    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=2,

    eval_metric="mlogloss",

    tree_method="hist",

    random_state=42,

    n_jobs=4
)


print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_train, y_train),
        (X_val, y_val)
    ],

    verbose=True
)


# =========================================================
# Test prediction
# =========================================================

probabilities = model.predict_proba(X_test)

predictions = np.argmax(
    probabilities,
    axis=1
)


# =========================================================
# Metrics
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        labels=np.arange(len(classes)),
        zero_division=0
    )
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
)

weighted_f1 = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )[2]
)


print("\n================================")
print("XGBOOST TEMPORAL TEST RESULTS")
print("================================")

print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")


print("\nPer-class results:")

for i, name in enumerate(classes):

    print(
        f"{name:30s} "
        f"P={precision[i]:.3f} "
        f"R={recall[i]:.3f} "
        f"F1={f1[i]:.3f} "
        f"Support={support[i]}"
    )


# =========================================================
# Save predictions
# =========================================================

np.savez(
    os.path.join(
        RESULT_DIR,
        "test_predictions.npz"
    ),

    y_true=y_test,
    y_pred=predictions,
    probabilities=probabilities
)


# =========================================================
# Save model
# =========================================================

model.save_model(
    os.path.join(
        RESULT_DIR,
        "xgboost_temporal.json"
    )
)


print("\nSaved results to:")
print(RESULT_DIR)