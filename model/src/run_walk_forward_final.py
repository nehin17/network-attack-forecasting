"""
Final walk-forward comparison for future_attack_type on existing CICIDS CSVs.

- Reads the same CSVs and selected features as build_forecasting_windows.py.
- Builds 10-minute histories and predicts the next minute's attack label.
- Uses expanding chronological folds within each capture; never randomly shuffles
  windows and never trains on future target times.
- Compares temporal-feature XGBoost with a GRU neural model.
- Saves reports to model/results/walk_forward_final/ and does not overwrite
  existing datasets or model artifacts.

Run from project root:
  .\.venv\Scripts\python.exe model\src\run_walk_forward_final.py
"""
import os
import sys
import csv
import random
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the GRU comparison. Activate the project's "
        ".venv and install/use the same PyTorch setup as the existing GRU script."
    ) from exc


BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE, "DATA", "processed", "CICIDS")
FEATURE_FILE = os.path.join(DATA_DIR, "cicids_selected_features.txt")
RESULT_DIR = os.path.join(BASE, "model", "results", "walk_forward_final")
HISTORY_MINUTES = 10
HORIZON_MINUTES = 1
N_FOLDS = 3
SEED = 42
EPOCHS = 35
BATCH_SIZE = 64
PATIENCE = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_label(label):
    label = label.strip()
    replacements = {
        "Web Attack Â Brute Force": "Web Attack - Brute Force",
        "Web Attack Â Sql Injection": "Web Attack - Sql Injection",
        "Web Attack Â XSS": "Web Attack - XSS",
        "Web Attack Â\x96 Brute Force": "Web Attack - Brute Force",
        "Web Attack Â\x96 Sql Injection": "Web Attack - Sql Injection",
        "Web Attack Â\x96 XSS": "Web Attack - XSS",
        "Web Attack – Brute Force": "Web Attack - Brute Force",
        "Web Attack – Sql Injection": "Web Attack - Sql Injection",
        "Web Attack – XSS": "Web Attack - XSS",
    }
    return replacements.get(label, label)


def parse_time(value):
    return datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S")


def load_capture(path, features):
    minute_values = defaultdict(lambda: defaultdict(list))
    minute_attacks = defaultdict(list)

    with open(path, "r", encoding="latin1", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Timestamp", "Label"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            return None
        available = set(reader.fieldnames)
        missing = [f for f in features if f not in available]
        if missing:
            raise ValueError(
                f"{os.path.basename(path)} is missing selected features: {missing[:8]}"
            )

        for row in reader:
            try:
                ts = parse_time(row.get("Timestamp", ""))
            except (ValueError, TypeError):
                continue
            label = row.get("Label", "")
            if not label:
                continue
            minute = ts.replace(second=0)
            for feature in features:
                try:
                    value = float(row.get(feature, ""))
                    if np.isfinite(value):
                        minute_values[minute][feature].append(value)
                except (ValueError, TypeError):
                    continue
            clean = normalize_label(label)
            if clean != "BENIGN":
                minute_attacks[minute].append(clean)

    timestamps = sorted(minute_values)
    if len(timestamps) < HISTORY_MINUTES + HORIZON_MINUTES + 10:
        return None

    X_minute, y_minute = [], []
    for ts in timestamps:
        vector = []
        for feature in features:
            vals = minute_values[ts].get(feature, [])
            vector.append(float(np.mean(vals)) if vals else 0.0)
        X_minute.append(vector)
        attacks = minute_attacks.get(ts, [])
        y_minute.append(Counter(attacks).most_common(1)[0][0] if attacks else "BENIGN")

    X_minute = np.nan_to_num(
        np.asarray(X_minute, dtype=np.float32),
        nan=0.0, posinf=0.0, neginf=0.0
    )

    # Every sample is indexed by the minute its target belongs to.
    samples, targets, target_idx = [], [], []
    for i in range(HISTORY_MINUTES, len(timestamps) - HORIZON_MINUTES + 1):
        samples.append(X_minute[i - HISTORY_MINUTES:i])
        future = y_minute[i:i + HORIZON_MINUTES]
        targets.append(next((lab for lab in future if lab != "BENIGN"), "BENIGN"))
        target_idx.append(i)

    return {
        "name": os.path.basename(path),
        "X": np.asarray(samples, dtype=np.float32),
        "y": np.asarray(targets, dtype=str),
        "target_idx": np.asarray(target_idx, dtype=np.int64),
        "n_minutes": len(timestamps),
    }


def temporal_features(X):
    return np.concatenate(
        [X[:, -1, :], X.mean(axis=1), X.std(axis=1), X[:, -1, :] - X[:, 0, :]],
        axis=1,
    ).astype(np.float32)


class GRUClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden=64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features, hidden_size=hidden, num_layers=1,
            batch_first=True, dropout=0.0
        )
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return self.head(self.dropout(output[:, -1, :]))


def make_fold_masks(captures, fold):
    """
    Expanding-window folds:
      fold 0: train targets before 50%, validate 50-60%, test 60-70%
      fold 1: train targets before 60%, validate 60-70%, test 70-80%
      fold 2: train targets before 70%, validate 70-80%, test 80-90%
    Percentages refer to target-time position within each capture.
    """
    train_parts, val_parts, test_parts = [], [], []
    train_cut = 0.50 + 0.10 * fold
    val_cut = train_cut + 0.10
    test_cut = val_cut + 0.10

    for cap in captures:
        denom = max(1, cap["n_minutes"] - 1)
        p = cap["target_idx"] / denom
        train_parts.append(p < train_cut)
        val_parts.append((p >= train_cut) & (p < val_cut))
        test_parts.append((p >= val_cut) & (p < test_cut))

    return train_parts, val_parts, test_parts


def gather(captures, masks):
    xs, ys = [], []
    for cap, mask in zip(captures, masks):
        if np.any(mask):
            xs.append(cap["X"][mask])
            ys.append(cap["y"][mask])
    if not xs:
        return np.empty((0, HISTORY_MINUTES, 0), np.float32), np.array([], dtype=str)
    return np.concatenate(xs), np.concatenate(ys)


def fit_xgb(Xtr, ytr, Xval, yval, Xte, classes):
    # XGBoost's sklearn wrapper expects contiguous labels in each fold.
    # A fold may not contain every global attack class in its training block.
    fold_classes = sorted(set(ytr.tolist()))
    fold_to_local = {label: i for i, label in enumerate(fold_classes)}
    local_to_global = np.asarray([class_to_id[label] for label in fold_classes], dtype=np.int64)
    train_ids = np.asarray([fold_to_local[x] for x in ytr], dtype=np.int64)
    # Validation examples of unseen-to-training classes cannot be scored by this
    # fold's model; filter them only for early-stopping loss, then score all test
    # examples normally (unseen classes necessarily cannot be predicted).
    val_mask = np.isin(yval, fold_classes)
    if not np.any(val_mask):
        raise ValueError("This fold has no validation examples from training-seen classes.")
    val_ids = np.asarray([fold_to_local[x] for x in yval[val_mask]], dtype=np.int64)
    sample_weights = compute_sample_weight("balanced", train_ids)
    model = XGBClassifier(
        objective="multi:softprob" if len(fold_classes) > 2 else "multi:softprob",
        num_class=len(fold_classes),
        n_estimators=500,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=4,
        early_stopping_rounds=25,
    )
    model.fit(
        temporal_features(Xtr), train_ids,
        sample_weight=sample_weights,
        eval_set=[(temporal_features(Xval[val_mask]), val_ids)],
        verbose=False,
    )
    local_predictions = model.predict(temporal_features(Xte)).astype(np.int64)
    global_predictions = local_to_global[local_predictions]
    return global_predictions, getattr(model, "best_iteration", None), model


def fit_gru(Xtr, ytr, Xval, yval, Xte, classes, fold):
    # Feature-wise standardization fitted only on training history values.
    flat = Xtr.reshape(-1, Xtr.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0

    def scale(x):
        return np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr_s, Xval_s, Xte_s = scale(Xtr), scale(Xval), scale(Xte)
    ytr_ids = np.asarray([class_to_id[x] for x in ytr], dtype=np.int64)
    yval_ids = np.asarray([class_to_id[x] for x in yval], dtype=np.int64)
    yte_dummy = None

    counts = np.bincount(ytr_ids, minlength=len(classes)).astype(np.float32)
    weights = np.zeros(len(classes), dtype=np.float32)
    present = counts > 0
    weights[present] = len(ytr_ids) / (present.sum() * counts[present])
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    seed_everything(SEED + fold)
    model = GRUClassifier(Xtr.shape[-1], len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_ds = TensorDataset(
        torch.tensor(Xtr_s, dtype=torch.float32),
        torch.tensor(ytr_ids, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    val_x = torch.tensor(Xval_s, dtype=torch.float32, device=DEVICE)
    val_y = torch.tensor(yval_ids, dtype=torch.long, device=DEVICE)
    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_x), val_y).item()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte_s, dtype=torch.float32, device=DEVICE))
        pred_ids = torch.argmax(logits, dim=1).cpu().numpy()
    return pred_ids, epoch + 1, model


def summarize(y_true, pred_ids, classes, model_name, fold, out_lines):
    y_ids = np.asarray([class_to_id[x] for x in y_true], dtype=np.int64)
    p, r, f, support = precision_recall_fscore_support(
        y_ids, pred_ids, labels=np.arange(len(classes)), zero_division=0
    )
    acc = accuracy_score(y_ids, pred_ids)
    macro_f1 = float(np.mean(f))
    weighted_f1 = precision_recall_fscore_support(
        y_ids, pred_ids, average="weighted", zero_division=0
    )[2]
    out_lines.append(
        f"{model_name} | fold {fold + 1} | n={len(y_true)} | "
        f"accuracy={acc:.4f} | macro_precision={np.mean(p):.4f} | "
        f"macro_recall={np.mean(r):.4f} | macro_f1={macro_f1:.4f} | "
        f"weighted_f1={weighted_f1:.4f}"
    )
    print(out_lines[-1])
    report = classification_report(
        y_ids, pred_ids, labels=np.arange(len(classes)),
        target_names=classes, zero_division=0, digits=3
    )
    out_lines.append(report)
    return {
        "accuracy": acc,
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "macro_f1": macro_f1,
        "weighted_f1": float(weighted_f1),
        "support": support,
    }


def main():
    global class_to_id
    print('[START] Walk-forward evaluation script started.', flush=True)
    print(f'[INFO] Project root: {BASE}', flush=True)
    print(f'[INFO] CSV directory: {DATA_DIR}', flush=True)
    print(f'[INFO] Output directory: {RESULT_DIR}', flush=True)
    seed_everything()
    os.makedirs(RESULT_DIR, exist_ok=True)

    if not os.path.isfile(FEATURE_FILE):
        raise FileNotFoundError(f"Selected-feature file not found: {FEATURE_FILE}")
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f if line.strip()]
    files = sorted(
        os.path.join(DATA_DIR, name)
        for name in os.listdir(DATA_DIR)
        if name.lower().endswith(".csv")
    )
    print(f"[INFO] Found {len(files)} CSV files.", flush=True)
    if not files:
        raise FileNotFoundError(
            f"No CSV captures found in {DATA_DIR}. This script needs the same source "
            "capture CSVs used by build_forecasting_windows.py."
        )

    captures = []
    for file_index, path in enumerate(files, start=1):
        print(f"[LOAD {file_index}/{len(files)}] Reading {os.path.basename(path)} ...", flush=True)
        cap = load_capture(path, features)
        if cap is not None:
            captures.append(cap)
            print(f"[LOADED] {cap['name']}: {cap['n_minutes']} minute bins, {len(cap['y'])} windows", flush=True)
    if not captures:
        raise RuntimeError("No eligible captures were loaded.")

    # Class universe is derived from actual processed target labels in all captures.
    classes = sorted(set(label for cap in captures for label in cap["y"]))
    class_to_id = {label: i for i, label in enumerate(classes)}
    print(f"\nClasses observed in source captures: {len(classes)}", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print("Three expanding chronological folds; original datasets remain untouched.\n", flush=True)

    summary_lines = [
        "FINAL WALK-FORWARD FUTURE ATTACK TYPE COMPARISON",
        f"History={HISTORY_MINUTES} minutes; horizon={HORIZON_MINUTES} minute",
        f"Classes={classes}",
        f"Device={DEVICE}",
        "",
    ]
    aggregate = {"Temporal XGBoost": [], "GRU": []}

    for fold in range(N_FOLDS):
        train_masks, val_masks, test_masks = make_fold_masks(captures, fold)
        Xtr, ytr = gather(captures, train_masks)
        Xval, yval = gather(captures, val_masks)
        Xte, yte = gather(captures, test_masks)

        print(f"\n{'=' * 24} FOLD {fold + 1}/{N_FOLDS} {'=' * 24}")
        print(f"Train={len(ytr)} Val={len(yval)} Test={len(yte)}", flush=True)
        print(f"Train classes={len(set(ytr))}; val classes={len(set(yval))}; test classes={len(set(yte))}", flush=True)
        summary_lines.append(
            f"\nFOLD {fold + 1}: train={len(ytr)}, val={len(yval)}, test={len(yte)}; "
            f"train_classes={len(set(ytr))}, val_classes={len(set(yval))}, "
            f"test_classes={len(set(yte))}"
        )

        if not len(ytr) or not len(yval) or not len(yte):
            msg = "Skipping fold because train, validation, or test block is empty."
            print(msg)
            summary_lines.append(msg)
            continue

        pred_xgb, best_iter, xgb_model = fit_xgb(Xtr, ytr, Xval, yval, Xte, classes)
        xgb_ids = pred_xgb
        xgb_result = summarize(yte, xgb_ids, classes, "Temporal XGBoost", fold, summary_lines)
        xgb_result["best_iteration"] = best_iter
        aggregate["Temporal XGBoost"].append(xgb_result)
        xgb_model.save_model(os.path.join(RESULT_DIR, f"xgboost_fold_{fold + 1}.json"))

        pred_gru, epochs, gru_model = fit_gru(Xtr, ytr, Xval, yval, Xte, classes, fold)
        gru_result = summarize(yte, pred_gru, classes, "GRU", fold, summary_lines)
        gru_result["epochs"] = epochs
        aggregate["GRU"].append(gru_result)
        torch.save(gru_model.state_dict(), os.path.join(RESULT_DIR, f"gru_fold_{fold + 1}.pt"))

        np.savez_compressed(
            os.path.join(RESULT_DIR, f"fold_{fold + 1}_predictions.npz"),
            y_true=yte,
            xgboost_pred=np.asarray([classes[i] for i in xgb_ids]),
            gru_pred=np.asarray([classes[i] for i in pred_gru]),
        )

    summary_lines.append("\nAGGREGATED FOLD MEANS (each fold equally weighted)")
    print("\nAGGREGATED FOLD MEANS")
    for model_name, rows in aggregate.items():
        if not rows:
            line = f"{model_name}: no successful folds"
        else:
            metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
            means = {m: float(np.mean([row[m] for row in rows])) for m in metrics}
            line = model_name + " | " + " | ".join(f"{m}={v:.4f}" for m, v in means.items())
        print(line)
        summary_lines.append(line)

    summary_lines += [
        "",
        "Interpretation notes:",
        "- Fold metrics are chronological and use a common test block per fold.",
        "- A class absent from a particular test block has zero F1 in that fold's fixed-class macro score.",
        "- The three fold means are descriptive, not a guarantee of future performance.",
        "- No random oversampling or synthetic test examples were used.",
    ]
    report_path = os.path.join(RESULT_DIR, "walk_forward_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSaved report and fold models to:\n{RESULT_DIR}", flush=True)
    print("Open walk_forward_report.txt for the full per-class results.", flush=True)


if __name__ == "__main__":
    main()
