import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_DIR = "DATA/processed/CICIDS"
RESULT_DIR = "model/results/lstm"

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
PATIENCE = 7

INPUT_SIZE = 70
HIDDEN_SIZE = 64
NUM_LAYERS = 1

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# MODEL
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.20)

        self.classifier = nn.Linear(
            hidden_size,
            num_classes
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = output[:, -1, :]

        last_output = self.dropout(
            last_output
        )

        return self.classifier(
            last_output
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(filename):

    path = os.path.join(
        DATA_DIR,
        filename
    )

    data = np.load(
        path,
        allow_pickle=True
    )

    return data["X"], data["y"]


# ============================================================
# TRAIN-ONLY STANDARDIZATION
# ============================================================

def fit_scaler(X):

    # Flatten samples and timesteps.
    flat = X.reshape(
        -1,
        X.shape[-1]
    )

    mean = flat.mean(
        axis=0
    )

    std = flat.std(
        axis=0
    )

    # Prevent division by zero.
    std[std < 1e-8] = 1.0

    return mean, std


def transform(X, mean, std):

    return (
        X - mean
    ) / std


# ============================================================
# TEMPERED CLASS WEIGHTS
# ============================================================

def build_class_weights(
    y,
    num_classes
):

    counts = np.bincount(
        y,
        minlength=num_classes
    )

    total = len(y)

    weights = []

    for count in counts:

        if count == 0:

            weights.append(0.0)

        else:

            # Square-root tempered inverse frequency.
            weight = np.sqrt(
                total /
                (num_classes * count)
            )

            # Prevent extreme gradients.
            weight = np.clip(
                weight,
                0.5,
                10.0
            )

            weights.append(
                weight
            )

    weights = np.asarray(
        weights,
        dtype=np.float32
    )

    return weights, counts


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    print(
        "\n=========================================="
    )

    print(
        "FINAL LSTM FORECASTING PIPELINE"
    )

    print(
        "=========================================="
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X_train, y_train_text = load_dataset(
        "train_multiclass.npz"
    )

    X_val, y_val_text = load_dataset(
        "val_multiclass.npz"
    )

    X_test, y_test_text = load_dataset(
        "test_multiclass.npz"
    )

    classes = np.load(
        os.path.join(
            DATA_DIR,
            "multiclass_classes.npy"
        ),
        allow_pickle=True
    ).tolist()

    class_to_index = {
        label: i
        for i, label in enumerate(classes)
    }

    y_train = np.asarray(
        [
            class_to_index[x]
            for x in y_train_text
        ],
        dtype=np.int64
    )

    y_val = np.asarray(
        [
            class_to_index[x]
            for x in y_val_text
        ],
        dtype=np.int64
    )

    y_test = np.asarray(
        [
            class_to_index[x]
            for x in y_test_text
        ],
        dtype=np.int64
    )

    num_classes = len(classes)

    print("\nDataset:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    print(
        "\nNumber of classes:",
        num_classes
    )

    # --------------------------------------------------------
    # SCALING
    # --------------------------------------------------------

    print(
        "\nFitting scaler on TRAINING DATA ONLY..."
    )

    mean, std = fit_scaler(
        X_train
    )

    X_train = transform(
        X_train,
        mean,
        std
    )

    X_val = transform(
        X_val,
        mean,
        std
    )

    X_test = transform(
        X_test,
        mean,
        std
    )

    # Save scaler.
    np.savez(
        os.path.join(
            RESULT_DIR,
            "lstm_scaler.npz"
        ),
        mean=mean,
        std=std
    )

    print(
        "Scaler saved."
    )

    print(
        "Training scaled mean:",
        f"{X_train.mean():.6f}"
    )

    print(
        "Training scaled std:",
        f"{X_train.std():.6f}"
    )

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------

    weights, counts = (
        build_class_weights(
            y_train,
            num_classes
        )
    )

    class_weights = torch.tensor(
        weights,
        dtype=torch.float32
    )

    print(
        "\nTraining class distribution:"
    )

    for i, label in enumerate(classes):

        print(
            f"{label:30s} "
            f"{counts[i]:5d} "
            f"weight={weights[i]:.3f}"
        )

    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train_tensor,
            y_train_tensor
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val_tensor,
            y_val_tensor
        ),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3
    )

    print(
        "\nModel:"
    )

    print(model)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float("inf")
    best_state = None

    patience_counter = 0

    print(
        "\n=========================================="
    )

    print(
        "TRAINING"
    )

    print(
        "=========================================="
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(
                X_batch
            )

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            # Stabilize recurrent training.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            train_loss += (
                loss.item()
                * X_batch.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                logits = model(
                    X_batch
                )

                loss = criterion(
                    logits,
                    y_batch
                )

                val_loss += (
                    loss.item()
                    * X_batch.size(0)
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                correct += (
                    predictions == y_batch
                ).sum().item()

                total += y_batch.size(0)

        val_loss /= len(
            val_loader.dataset
        )

        val_accuracy = (
            correct / total
        )

        scheduler.step(
            val_loss
        )

        history[
            "train_loss"
        ].append(train_loss)

        history[
            "val_loss"
        ].append(val_loss)

        history[
            "val_accuracy"
        ].append(val_accuracy)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "\nEarly stopping."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    checkpoint = {
        "model_state_dict":
            model.state_dict(),

        "classes":
            np.asarray(classes),

        "input_size":
            INPUT_SIZE,

        "hidden_size":
            HIDDEN_SIZE,

        "num_layers":
            NUM_LAYERS,

        "num_classes":
            num_classes
    }

    model_path = os.path.join(
        RESULT_DIR,
        "lstm_final.pt"
    )

    torch.save(
        checkpoint,
        model_path
    )

    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_test_tensor),
            BATCH_SIZE
        ):

            batch = X_test_tensor[
                start:start + BATCH_SIZE
            ]

            logits = model(
                batch
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            preds = torch.argmax(
                probs,
                dim=1
            )

            predictions.extend(
                preds.numpy()
            )

            probabilities.extend(
                probs.numpy()
            )

    y_pred = np.asarray(
        predictions
    )

    probabilities = np.asarray(
        probabilities
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "FINAL LSTM TEST RESULTS"
    )

    print(
        "=========================================="
    )

    print(
        f"Accuracy:           {accuracy:.4f}"
    )

    print(
        f"Macro Precision:    {macro_precision:.4f}"
    )

    print(
        f"Macro Recall:       {macro_recall:.4f}"
    )

    print(
        f"Macro F1:           {macro_f1:.4f}"
    )

    print(
        f"Weighted Precision: {weighted_precision:.4f}"
    )

    print(
        f"Weighted Recall:    {weighted_recall:.4f}"
    )

    print(
        f"Weighted F1:        {weighted_f1:.4f}"
    )

    print(
        "\nPer-class results:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(num_classes)),
            target_names=classes,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # SAVE NUMERIC RESULTS
    # --------------------------------------------------------

    np.savez(
        os.path.join(
            RESULT_DIR,
            "lstm_predictions.npz"
        ),
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        classes=np.asarray(classes)
    )

    # ========================================================
    # GRAPH 1 — TRAINING CURVES
    # ========================================================

    epochs_used = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs_used,
        history["train_loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs_used,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-Entropy Loss"
    )

    plt.title(
        "LSTM Training and Validation Loss"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "training_loss.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # GRAPH 2 — VALIDATION ACCURACY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        epochs_used,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "LSTM Validation Accuracy"
    )

    plt.ylim(
        0,
        1
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "validation_accuracy.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # GRAPH 3 — CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(num_classes))
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=classes
    )

    display.plot(
        ax=ax,
        xticks_rotation=75,
        cmap="Blues",
        colorbar=True
    )

    ax.set_title(
        "LSTM Future Attack Type — Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "confusion_matrix.png"
        ),
        dpi=250
    )

    plt.close()

    # ========================================================
    # GRAPH 4 — TEST CLASS DISTRIBUTION
    # ========================================================

    unique, counts = np.unique(
        y_test,
        return_counts=True
    )

    labels_present = [
        classes[i]
        for i in unique
    ]

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        labels_present,
        counts
    )

    plt.xlabel(
        "Actual Class"
    )

    plt.ylabel(
        "Number of Test Samples"
    )

    plt.title(
        "LSTM Test Set Class Distribution"
    )

    plt.xticks(
        rotation=65,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULT_DIR,
            "test_class_distribution.png"
        ),
        dpi=200
    )

    plt.close()

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n=========================================="
    )

    print(
        "LSTM PIPELINE COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Results: {RESULT_DIR}"
    )

    print(
        "\nGenerated:"
    )

    print(
        "  training_loss.png"
    )

    print(
        "  validation_accuracy.png"
    )

    print(
        "  confusion_matrix.png"
    )

    print(
        "  test_class_distribution.png"
    )

    print(
        "  lstm_predictions.npz"
    )

    print(
        "  lstm_scaler.npz"
    )


if __name__ == "__main__":
    main()