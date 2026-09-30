import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(
    BASE, "DATA", "processed", "CICIDS"
)

RESULT_DIR = os.path.join(
    BASE, "model", "results", "stacked_lstm"
)

os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# -----------------------------
# Load original frozen dataset
# -----------------------------

train = np.load(
    os.path.join(DATA_DIR, "train_multiclass.npz")
)

val = np.load(
    os.path.join(DATA_DIR, "val_multiclass.npz")
)

test = np.load(
    os.path.join(DATA_DIR, "test_multiclass.npz")
)

classes = np.load(
    os.path.join(DATA_DIR, "multiclass_classes.npy"),
    allow_pickle=True
)

class_to_id = {
    name: i
    for i, name in enumerate(classes)
}

X_train = train["X"].astype(np.float32)
X_val = val["X"].astype(np.float32)
X_test = test["X"].astype(np.float32)

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

print("Train:", X_train.shape)
print("Val:", X_val.shape)
print("Test:", X_test.shape)


# -----------------------------
# Scale using TRAIN ONLY
# -----------------------------

mean = X_train.mean(axis=(0, 1))
std = X_train.std(axis=(0, 1))

std[std < 1e-8] = 1.0

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std


# -----------------------------
# Class weights
# -----------------------------

counts = np.bincount(
    y_train,
    minlength=len(classes)
)

weights = np.sqrt(
    len(y_train) / np.maximum(counts, 1)
)

weights = np.clip(
    weights,
    0.5,
    10.0
)

weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE
)


# -----------------------------
# DataLoaders
# -----------------------------

train_ds = TensorDataset(
    torch.tensor(X_train),
    torch.tensor(y_train)
)

val_ds = TensorDataset(
    torch.tensor(X_val),
    torch.tensor(y_val)
)

test_ds = TensorDataset(
    torch.tensor(X_test),
    torch.tensor(y_test)
)

train_loader = DataLoader(
    train_ds,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_ds,
    batch_size=128,
    shuffle=False
)


# -----------------------------
# Stacked LSTM
# -----------------------------

class StackedLSTM(nn.Module):

    def __init__(
        self,
        input_size=70,
        hidden_size=64,
        num_classes=15
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.25
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        x = output[:, -1, :]

        x = self.dropout(x)

        return self.classifier(x)


model = StackedLSTM(
    input_size=70,
    hidden_size=64,
    num_classes=len(classes)
).to(DEVICE)

print("\nModel:")
print(model)


# -----------------------------
# Training
# -----------------------------

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

best_val_loss = float("inf")
bad_epochs = 0
patience = 7

for epoch in range(40):

    model.train()

    running_loss = 0.0

    for X, y in train_loader:

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        output = model(X)

        loss = criterion(output, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        running_loss += (
            loss.item() * X.size(0)
        )

    train_loss = (
        running_loss / len(train_ds)
    )


    # Validation

    model.eval()

    val_loss = 0.0
    correct = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            output = model(X)

            loss = criterion(output, y)

            val_loss += (
                loss.item() * X.size(0)
            )

            correct += (
                output.argmax(1) == y
            ).sum().item()

    val_loss /= len(val_ds)

    val_acc = (
        correct / len(val_ds)
    )

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"train {train_loss:.4f} | "
        f"val {val_loss:.4f} | "
        f"val acc {val_acc:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                RESULT_DIR,
                "stacked_lstm_best.pt"
            )
        )

    else:

        bad_epochs += 1

        if bad_epochs >= patience:

            print("Early stopping.")
            break


# -----------------------------
# Test
# -----------------------------

model.load_state_dict(
    torch.load(
        os.path.join(
            RESULT_DIR,
            "stacked_lstm_best.pt"
        ),
        map_location=DEVICE
    )
)

model.eval()

predictions = []
probabilities = []
true_labels = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        output = model(X)

        probs = torch.softmax(
            output,
            dim=1
        )

        predictions.extend(
            probs.argmax(1).cpu().numpy()
        )

        probabilities.extend(
            probs.cpu().numpy()
        )

        true_labels.extend(
            y.numpy()
        )

predictions = np.array(predictions)
probabilities = np.array(probabilities)
true_labels = np.array(true_labels)


# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )
)

weighted_f1 = precision_recall_fscore_support(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)[2]


print("\n================================")
print("STACKED LSTM TEST RESULTS")
print("================================")

print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")
print(f"Macro F1:        {macro_f1:.4f}")
print(f"Weighted F1:     {weighted_f1:.4f}")


# -----------------------------
# Save
# -----------------------------

np.savez(
    os.path.join(
        RESULT_DIR,
        "test_predictions.npz"
    ),
    y_true=true_labels,
    y_pred=predictions,
    probabilities=probabilities
)

print("\nSaved results to:")
print(RESULT_DIR)