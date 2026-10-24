
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA = Path("DATA/raw/TON_IoT/Network/train_test_network.csv")
OUT = Path("model/results/ton_iot")
OUT.mkdir(parents=True, exist_ok=True)

print("Loading TON_IoT...")
df = pd.read_csv(DATA, low_memory=False)

# Target is attack category. Exclude binary label and IP identities.
y = df["type"].astype(str)
X = df.drop(
    columns=["type", "label", "src_ip", "dst_ip"],
    errors="ignore",
).copy()

# Group identical model-input patterns, even if source IP differs.
# Hash raw feature values before any learned preprocessing.
fingerprints = pd.util.hash_pandas_object(X, index=False)
groups, _ = pd.factorize(fingerprints, sort=False)

audit = pd.DataFrame({"group": groups, "target": y})
label_counts = audit.groupby("group")["target"].nunique()
conflicting_groups = int((label_counts > 1).sum())

print(f"Rows: {len(df):,}")
print(f"Input features: {X.shape[1]}")
print(f"Unique feature patterns: {len(np.unique(groups)):,}")
print(f"Patterns with conflicting labels: {conflicting_groups:,}")
print("\nClass distribution:")
print(y.value_counts().to_string())

classes = sorted(y.unique())
class_to_id = {name: i for i, name in enumerate(classes)}
y_id = y.map(class_to_id).to_numpy()
n_classes = len(classes)

cv = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

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
        num_class=n_classes,
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

all_results = []
reports = {}
start_all = time.time()

for model_name, prototype in models.items():
    print(f"\n{'=' * 60}\n{model_name}: grouped 5-fold evaluation")
    oof_pred = np.full(len(df), -1, dtype=int)
    fold_rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y_id, groups), start=1):
        y_tr, y_te = y_id[tr], y_id[te]
        train_classes = set(y_tr)
        test_classes = set(y_te)

        if len(train_classes) != n_classes:
            missing = sorted(set(range(n_classes)) - train_classes)
            raise RuntimeError(
                f"{model_name}, fold {fold}: training lacks classes "
                f"{[classes[i] for i in missing]}. "
                "This evaluation is inconclusive; do not report a winner."
            )

        # Assert no feature fingerprint crosses the split.
        assert set(groups[tr]).isdisjoint(set(groups[te]))

        Xtr = X.iloc[tr].copy()
        Xte = X.iloc[te].copy()

        cat_cols = Xtr.select_dtypes(
            include=["object", "str", "category"]
        ).columns.tolist()
        num_cols = [c for c in Xtr.columns if c not in cat_cols]

        for col in cat_cols:
            Xtr[col] = Xtr[col].fillna("__MISSING__").astype(str)
            Xte[col] = Xte[col].fillna("__MISSING__").astype(str)

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
        if cat_cols:
            Xtr[cat_cols] = encoder.fit_transform(Xtr[cat_cols])
            Xte[cat_cols] = encoder.transform(Xte[cat_cols])

        for col in num_cols:
            Xtr[col] = pd.to_numeric(Xtr[col], errors="coerce")
            Xte[col] = pd.to_numeric(Xte[col], errors="coerce")

        imputer = SimpleImputer(strategy="median")
        Xtr = imputer.fit_transform(Xtr)
        Xte = imputer.transform(Xte)

        # Fresh model for each fold.
        model = prototype.__class__(**prototype.get_params())
        start = time.time()
        model.fit(Xtr, y_tr)
        pred = model.predict(Xte).astype(int)
        elapsed = round(time.time() - start, 2)
        oof_pred[te] = pred

        fold_f1 = precision_recall_fscore_support(
            y_te, pred, labels=list(range(n_classes)),
            average="macro", zero_division=0
        )[2]

        fold_rows.append({
            "fold": fold,
            "train_rows": len(tr),
            "test_rows": len(te),
            "test_classes_present": len(test_classes),
            "macro_f1": float(fold_f1),
            "train_seconds": elapsed,
        })
        print(
            f"Fold {fold}/5: rows={len(te):,}, "
            f"classes={len(test_classes)}/{n_classes}, "
            f"macro_F1={fold_f1:.4f}, time={elapsed}s"
        )

    assert np.all(oof_pred >= 0)

    p, r, f1, support = precision_recall_fscore_support(
        y_id, oof_pred, labels=list(range(n_classes)),
        zero_division=0
    )
    cm = confusion_matrix(y_id, oof_pred, labels=list(range(n_classes)))

    # Normal is the benign class. FPR = benign rows predicted as attack / benign rows.
    normal_id = class_to_id.get("normal")
    if normal_id is not None:
        normal_total = cm[normal_id].sum()
        false_positive_rate = (
            (normal_total - cm[normal_id, normal_id]) / normal_total
            if normal_total else None
        )
    else:
        false_positive_rate = None

    result = {
        "output": "attack_type",
        "dataset": "TON_IoT",
        "model": model_name,
        "evaluation": "5-fold grouped by model-input fingerprint",
        "accuracy": float(accuracy_score(y_id, oof_pred)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": float(np.mean(f1)),
        "normal_false_positive_rate": (
            float(false_positive_rate)
            if false_positive_rate is not None else None
        ),
        "conflicting_feature_patterns": conflicting_groups,
        "folds": fold_rows,
        "per_class": {
            name: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
        "confusion_matrix": cm.tolist(),
    }
    all_results.append(result)
    reports[model_name] = result

    print("\nOut-of-fold metrics:")
    print(json.dumps({
        k: result[k] for k in [
            "accuracy", "macro_precision", "macro_recall",
            "macro_f1", "normal_false_positive_rate"
        ]
    }, indent=2))
    print("\nPer-class results:")
    print(classification_report(
        y_id, oof_pred, labels=list(range(n_classes)),
        target_names=classes, zero_division=0, digits=4
    ))

with open(OUT / "final_robustness_report.json", "w") as f:
    json.dump({
        "rows": len(df),
        "features": X.shape[1],
        "unique_input_patterns": int(len(np.unique(groups))),
        "conflicting_feature_patterns": conflicting_groups,
        "results": all_results,
        "elapsed_seconds": round(time.time() - start_all, 2),
    }, f, indent=2)

pd.DataFrame([
    {
        "model": r["model"],
        "accuracy": r["accuracy"],
        "macro_precision": r["macro_precision"],
        "macro_recall": r["macro_recall"],
        "macro_f1": r["macro_f1"],
        "normal_false_positive_rate": r["normal_false_positive_rate"],
    }
    for r in all_results
]).to_csv(OUT / "final_robustness_comparison.csv", index=False)

print("\nSaved:")
print(OUT / "final_robustness_report.json")
print(OUT / "final_robustness_comparison.csv")
print("\nThis evaluates classification robustness, not future forecasting.")
