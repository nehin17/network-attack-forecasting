import numpy as np
import torch

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from lstm_model import LSTMClassifier


DATA_DIR = "DATA/processed/CICIDS"


# -----------------------------
# Load validation data
# -----------------------------

data = np.load(
    f"{DATA_DIR}/val.npz"
)

X_val = torch.tensor(
    data["X"],
    dtype=torch.float32
)

y_val = data["y"].astype(int)


# -----------------------------
# Load trained LSTM
# -----------------------------

checkpoint = torch.load(
    f"{DATA_DIR}/lstm_model.pt",
    map_location="cpu",
    weights_only=False
)

model = LSTMClassifier(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    num_layers=checkpoint["num_layers"],
    num_classes=2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# -----------------------------
# Get attack probabilities
# -----------------------------

with torch.no_grad():

    outputs = model(X_val)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    attack_probability = probabilities[:, 1].numpy()


# -----------------------------
# Test thresholds
# -----------------------------

thresholds = np.arange(
    0.10,
    0.51,
    0.05
)

print("\n==============================")
print("VALIDATION THRESHOLD ANALYSIS")
print("==============================")

print(
    "\nThreshold | Precision | Recall | F1"
)

print("------------------------------------")


best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:

    predictions = (
        attack_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    print(
        f"{threshold:9.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print("\n==============================")
print("BEST VALIDATION THRESHOLD")
print("==============================")

print(f"Threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1:.4f}")