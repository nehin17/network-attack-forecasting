import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE, "DATA", "processed", "CICIDS")
RESULT_DIR = os.path.join(BASE, "model", "results", "xgboost_weighted")
os.makedirs(RESULT_DIR, exist_ok=True)

# Use the chronological test set, not the small coverage sanity-check set.
train = np.load(os.path.join(DATA_DIR, "train_multiclass.npz"))
val = np.load(os.path.join(DATA_DIR, "val_multiclass.npz"))
test = np.load(os.path.join(DATA_DIR, "test_multiclass.npz"))
classes = np.load(
    os.path.join(DATA_DIR, "multiclass_classes.npy"),
    allow_pickle=True,
)

class_to_id = {name: i for i, name in enumerate(classes)}

X_train_raw = train["X"].astype(np.float32)
X_val_raw = val["X"].astype(np.float32)
X_test_raw = test["X"].astype(np.float32)

y_train = np.array([class_to_id[x] for x in train["y"]], dtype=np.int64)
y_val = np.array([class_to_id[x] for x in val["y"]], dtype=np.int64)
y_test = np.array([class_to_id[x] for x in test["y"]], dtype=np.int64)


def make_temporal_features(X):
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    delta = X[:, -1, :] - X[:, 0, :]
    return np.concatenate([last, mean, std, delta], axis=1)


X_train = make_temporal_features(X_train_raw)
X_val = make_temporal_features(X_val_raw)
X_test = make_temporal_features(X_test_raw)

print("Original sequence shapes:")
print(f"Train: {X_train_raw.shape}")
print(f"Val:   {X_val_raw.shape}")
print(f"Test:  {X_test_raw.shape}")
print(f"Classes: {len(classes)}")

print("\nTemporal feature shapes:")
print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

# Balance the training examples by their observed class frequencies.
sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

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
    n_jobs=4,
    early_stopping_rounds=20,
)

print("\nTraining class-weighted Temporal XGBoost with early stopping...")
model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
    eval_set=[(X_val, y_val)],
    verbose=True,
)

probabilities = model.predict_proba(X_test)
predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)
precision, recall, f1, support = precision_recall_fscore_support(
    y_test,
    predictions,
    labels=np.arange(len(classes)),
    zero_division=0,
)

macro_precision = float(np.mean(precision))
macro_recall = float(np.mean(recall))
macro_f1 = float(np.mean(f1))
weighted_f1 = precision_recall_fscore_support(
    y_test, predictions, average="weighted", zero_division=0
)[2]

print("\n================================")
print("WEIGHTED TEMPORAL XGBOOST RESULTS")
print("================================")
print(f"Best iteration:  {getattr(model, 'best_iteration', 'N/A')}")
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")

print("\nPer-class results:")
for i, class_name in enumerate(classes):
    print(
        f"{str(class_name):30s} "
        f"P={precision[i]:.3f} "
        f"R={recall[i]:.3f} "
        f"F1={f1[i]:.3f} "
        f"Support={support[i]}"
    )

np.savez(
    os.path.join(RESULT_DIR, "test_predictions.npz"),
    y_true=y_test,
    y_pred=predictions,
    probabilities=probabilities,
)
model.save_model(os.path.join(RESULT_DIR, "xgboost_temporal_weighted.json"))

with open(os.path.join(RESULT_DIR, "metrics.txt"), "w", encoding="utf-8") as f:
    f.write("Weighted Temporal XGBoost with early stopping\n")
    f.write(f"Best iteration: {getattr(model, 'best_iteration', 'N/A')}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Macro Precision: {macro_precision:.4f}\n")
    f.write(f"Macro Recall: {macro_recall:.4f}\n")
    f.write(f"Macro F1: {macro_f1:.4f}\n")
    f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
    f.write("Per-class results:\n")
    for i, class_name in enumerate(classes):
        f.write(
            f"{str(class_name):30s} "
            f"P={precision[i]:.3f} R={recall[i]:.3f} "
            f"F1={f1[i]:.3f} Support={support[i]}\n"
        )

print(f"\nSaved weighted experiment separately to:\n{RESULT_DIR}")
