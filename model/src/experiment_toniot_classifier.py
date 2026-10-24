
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("DATA/raw/TON_IoT/Network/train_test_network.csv")
RESULTS_DIR = Path("model/results/ton_iot")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading TON_IoT...")
df = pd.read_csv(DATA_PATH, low_memory=False)

# Remove exact duplicate rows for this initial benchmark.
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows: {before:,} -> {len(df):,} after deduplication")

# Multiclass target. Do not include either label column as an input.
target = "type"
drop_cols = [
    "type", "label", "src_ip", "dst_ip"
]
X = df.drop(columns=drop_cols, errors="ignore").copy()
y = df[target].astype(str)

# Encode categorical features using training data only.
cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Clean and encode categorical features; unseen test categories get a safe code.
X_train = X_train.copy()
X_test = X_test.copy()

for col in cat_cols:
    X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
    X_test[col] = X_test[col].fillna("__MISSING__").astype(str)

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)
if cat_cols:
    X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = encoder.transform(X_test[cat_cols])

for col in num_cols:
    X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X.columns,
)
X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X.columns,
)

classes = sorted(y_train.unique())
class_to_id = {name: i for i, name in enumerate(classes)}
y_train_id = y_train.map(class_to_id).to_numpy()
y_test_id = y_test.map(class_to_id).to_numpy()

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=250,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    ),
}

summary = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start = time.time()
    model.fit(X_train, y_train_id)
    elapsed = time.time() - start

    pred = model.predict(X_test).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_id, pred, average="macro", zero_division=0
    )

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test_id, pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "train_seconds": round(elapsed, 2),
    }
    summary.append(result)

    print(json.dumps(result, indent=2))
    print("\nPer-class report:")
    print(classification_report(
        y_test_id,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        zero_division=0,
        digits=4,
    ))

    if name == "XGBoost":
        model.save_model(str(RESULTS_DIR / "toniot_xgboost_baseline.json"))

pd.DataFrame(summary).to_csv(
    RESULTS_DIR / "baseline_comparison.csv", index=False
)
print("\nSaved comparison to:", RESULTS_DIR / "baseline_comparison.csv")
print(pd.DataFrame(summary).to_string(index=False))
