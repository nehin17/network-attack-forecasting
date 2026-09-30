import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt


# -----------------------------
# Paths
# -----------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(BASE, "DATA", "processed", "CICIDS")
RESULT_DIR = os.path.join(BASE, "model", "results", "cnn_lstm")
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# -----------------------------
# Load frozen dataset
# -----------------------------
train = np.load(os.path.join(DATA_DIR, "train_multiclass.npz"))
val = np.load(os.path.join(DATA_DIR, "val_multiclass.npz"))
test = np.load(os.path.join(DATA_DIR, "test_multiclass.npz"))

X_train = train["X"].astype(np.float32)
X_val = val["X"].astype(np.float32)
X_test = test["X"].astype(np.float32)

classes = np.load(
    os.path.join(DATA_DIR, "multiclass_classes.npy"),
    allow_pickle=True
)

class_to_id = {
    name: i for i, name in enumerate(classes)
}

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
print("Classes:", len(classes))


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
counts = np.bincount(y_train, minlength=len(classes))

weights = np.sqrt(
    len(y_train) / np.maximum(counts, 1)
)

weights = np.clip(weights, 0.5, 10.0)

weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")
for i, name in enumerate(classes):
    print(f"{name}: {weights[i].item():.3f}")


# -----------------------------
# Dataset
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
# CNN + LSTM
# -----------------------------
class CNNLSTM(nn.Module):

    def __init__(
        self,
        input_size=70,
        num_classes=15
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            nn.Dropout(0.2)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        # x = [batch, time, features]

        x = x.transpose(1, 2)

        # [batch, features, time]
        x = self.conv(x)

        # [batch, time, channels]
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)

        x = x[:, -1, :]

        x = self.dropout(x)

        return self.classifier(x)


model = CNNLSTM(
    input_size=X_train.shape[2],
    num_classes=len(classes)
).to(DEVICE)

print("\nModel:")
print(model)


# -----------------------------
# Training
# -----------------------------
criterion = nn.CrossEntropyLoss(weight=weights)

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
patience = 7
bad_epochs = 0

train_losses = []
val_losses = []
val_accuracies = []


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

        running_loss += loss.item() * X.size(0)

    train_loss = running_loss / len(train_ds)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()

    val_loss = 0.0
    correct = 0

    with torch.no_grad():

        for X, y in val_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            output = model(X)

            loss = criterion(output, y)

            val_loss += loss.item() * X.size(0)

            pred = output.argmax(dim=1)

            correct += (pred == y).sum().item()

    val_loss /= len(val_ds)
    val_acc = correct / len(val_ds)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch + 1:02d} | "
        f"train {train_loss:.4f} | "
        f"val {val_loss:.4f} | "
        f"val acc {val_acc:.4f}"
    )

    # -------------------------
    # Early stopping
    # -------------------------
    if val_loss < best_val_loss:

        best_val_loss = val_loss
        bad_epochs = 0

        torch.save(
            model.state_dict(),
            os.path.join(
                RESULT_DIR,
                "cnn_lstm_best.pt"
            )
        )

    else:

        bad_epochs += 1

        if bad_epochs >= patience:

            print("Early stopping.")
            break


# -----------------------------
# Load best model
# -----------------------------
model.load_state_dict(
    torch.load(
        os.path.join(
            RESULT_DIR,
            "cnn_lstm_best.pt"
        ),
        map_location=DEVICE
    )
)

model.eval()


# -----------------------------
# Test
# -----------------------------
all_preds = []
all_probs = []
all_true = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        output = model(X)

        probs = torch.softmax(output, dim=1)

        preds = probs.argmax(dim=1)

        all_preds.extend(
            preds.cpu().numpy()
        )

        all_probs.extend(
            probs.cpu().numpy()
        )

        all_true.extend(
            y.numpy()
        )

all_preds = np.array(all_preds)
all_probs = np.array(all_probs)
all_true = np.array(all_true)


# -----------------------------
# Metrics
# -----------------------------
accuracy = accuracy_score(
    all_true,
    all_preds
)

precision, recall, f1, support = precision_recall_fscore_support(
    all_true,
    all_preds,
    labels=np.arange(len(classes)),
    zero_division=0
)

macro_p = precision_recall_fscore_support(
    all_true,
    all_preds,
    average="macro",
    zero_division=0
)[0]

macro_r = precision_recall_fscore_support(
    all_true,
    all_preds,
    average="macro",
    zero_division=0
)[1]

macro_f1 = precision_recall_fscore_support(
    all_true,
    all_preds,
    average="macro",
    zero_division=0
)[2]

weighted_f1 = precision_recall_fscore_support(
    all_true,
    all_preds,
    average="weighted",
    zero_division=0
)[2]


print("\n==============================")
print("CNN + LSTM TEST RESULTS")
print("==============================")

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {macro_p:.4f}")
print(f"Macro Recall:      {macro_r:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")

print("\nPer-class results:")

for i, name in enumerate(classes):

    print(
        f"{name:30s} "
        f"P={precision[i]:.3f} "
        f"R={recall[i]:.3f} "
        f"F1={f1[i]:.3f} "
        f"Support={support[i]}"
    )


# -----------------------------
# Save predictions
# -----------------------------
np.savez(
    os.path.join(
        RESULT_DIR,
        "test_predictions.npz"
    ),
    y_true=all_true,
    y_pred=all_preds,
    probabilities=all_probs
)


# -----------------------------
# Graph 1: training loss
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    train_losses,
    label="Training Loss"
)

plt.plot(
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN + LSTM Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "training_loss.png"
    ),
    dpi=200
)

plt.close()


# -----------------------------
# Graph 2: validation accuracy
# -----------------------------
plt.figure(figsize=(9, 5))

plt.plot(
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN + LSTM Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "validation_accuracy.png"
    ),
    dpi=200
)

plt.close()


# -----------------------------
# Graph 3: confusion matrix
# -----------------------------
cm = confusion_matrix(
    all_true,
    all_preds,
    labels=np.arange(len(classes))
)

plt.figure(figsize=(12, 10))

plt.imshow(cm)

plt.xticks(
    np.arange(len(classes)),
    classes,
    rotation=90
)

plt.yticks(
    np.arange(len(classes)),
    classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CNN + LSTM Confusion Matrix")

plt.colorbar()

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


print("\nSaved results to:")
print(RESULT_DIR)
