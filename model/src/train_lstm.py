import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from lstm_model import LSTMClassifier


DATA_DIR = "DATA/processed/CICIDS"
MODEL_DIR = "model"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

INPUT_SIZE = 70
HIDDEN_SIZE = 64
NUM_LAYERS = 1


def load_data(filename):
    data = np.load(
        os.path.join(DATA_DIR, filename),
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]

    return X, y


def main():

    print("Loading multiclass forecasting data...")

    X_train, y_train_text = load_data(
        "train_multiclass.npz"
    )

    X_val, y_val_text = load_data(
        "val_multiclass.npz"
    )

    # ----------------------------------------
    # Load exact class order used by dataset
    # ----------------------------------------

    classes = np.load(
        os.path.join(
            DATA_DIR,
            "multiclass_classes.npy"
        ),
        allow_pickle=True
    ).tolist()

    class_to_index = {
        label: index
        for index, label in enumerate(classes)
    }

    y_train = np.asarray(
        [
            class_to_index[label]
            for label in y_train_text
        ],
        dtype=np.int64
    )

    y_val = np.asarray(
        [
            class_to_index[label]
            for label in y_val_text
        ],
        dtype=np.int64
    )

    num_classes = len(classes)

    print("\nDataset:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)

    print("\nClasses:")
    for i, label in enumerate(classes):
        print(f"{i}: {label}")

    # ----------------------------------------
    # Tensor conversion
    # ----------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    val_dataset = TensorDataset(
        X_val,
        y_val
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # ----------------------------------------
    # Class weights
    # ----------------------------------------
    #
    # Calculated ONLY from training data.
    # ----------------------------------------

    class_counts = np.bincount(
        y_train.numpy(),
        minlength=num_classes
    )

    total = len(y_train)

    class_weights = []

    for count in class_counts:

        if count == 0:
            class_weights.append(0.0)
        else:
            class_weights.append(
                total / (
                    num_classes * count
                )
            )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    )

    print("\nTraining class counts:")

    for i, count in enumerate(class_counts):
        print(
            f"{classes[i]}: {count}"
        )

    print("\nClass weights:")
    print(class_weights)

    # ----------------------------------------
    # Model
    # ----------------------------------------

    model = LSTMClassifier(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nModel:")
    print(model)

    # ----------------------------------------
    # Training
    # ----------------------------------------

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0

        for X_batch, y_batch in train_loader:

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * X_batch.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        # ------------------------------------
        # Validation
        # ------------------------------------

        model.eval()

        val_loss = 0.0

        correct = 0
        total_val = 0

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                logits = model(X_batch)

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

                total_val += y_batch.size(0)

        val_loss /= len(
            val_loader.dataset
        )

        val_accuracy = (
            correct / total_val
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f}"
        )

        # ------------------------------------
        # Save best validation model
        # ------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    # ----------------------------------------
    # Save checkpoint
    # ----------------------------------------

    checkpoint = {
        "model_state_dict": best_state,
        "classes": np.asarray(
            classes
        ),
        "input_size": INPUT_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "num_classes": num_classes
    }

    output_path = os.path.join(
        MODEL_DIR,
        "lstm_multiclass.pt"
    )

    torch.save(
        checkpoint,
        output_path
    )

    print(
        "\n================================"
    )

    print(
        "MULTICLASS LSTM TRAINING COMPLETE"
    )

    print(
        "================================"
    )

    print(
        "Best validation loss:",
        f"{best_val_loss:.4f}"
    )

    print(
        "Saved:",
        output_path
    )


if __name__ == "__main__":
    main()