import os
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from lstm_model import LSTMClassifier


DATA_DIR = "DATA/processed/CICIDS"
MODEL_PATH = "model/lstm_multiclass.pt"

BATCH_SIZE = 32


def main():

    print("Loading test data...")

    data = np.load(
        os.path.join(
            DATA_DIR,
            "test_multiclass.npz"
        ),
        allow_pickle=True
    )

    X_test = data["X"]
    y_test_text = data["y"]

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False
    )

    classes = checkpoint["classes"].tolist()

    class_to_index = {
        label: i
        for i, label in enumerate(classes)
    }

    y_test = np.asarray(
        [
            class_to_index[label]
            for label in y_test_text
        ],
        dtype=np.int64
    )

    # ----------------------------------------
    # Model
    # ----------------------------------------

    model = LSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=checkpoint["num_classes"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # ----------------------------------------
    # Prediction
    # ----------------------------------------

    X_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for start in range(
            0,
            len(X_tensor),
            BATCH_SIZE
        ):

            batch = X_tensor[
                start:start + BATCH_SIZE
            ]

            logits = model(batch)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_predictions.extend(
                predictions.numpy()
            )

            all_probabilities.extend(
                probabilities.numpy()
            )

    y_pred = np.asarray(
        all_predictions
    )

    probabilities = np.asarray(
        all_probabilities
    )

    # ----------------------------------------
    # Overall metrics
    # ----------------------------------------

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

    print("\n================================")
    print("LSTM MULTICLASS TEST RESULTS")
    print("================================")

    print(
        f"Test samples: {len(y_test)}"
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

    # ----------------------------------------
    # Per-class results
    # ----------------------------------------

    print("\n================================")
    print("PER-CLASS RESULTS")
    print("================================")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(classes))),
            target_names=classes,
            zero_division=0
        )
    )

    # ----------------------------------------
    # Confusion matrix
    # ----------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(classes)))
    )

    print("\n================================")
    print("CONFUSION MATRIX")
    print("================================")

    print(
        "Rows = actual"
    )

    print(
        "Columns = predicted"
    )

    print()

    print(
        "      " +
        " ".join(
            f"{i:4d}"
            for i in range(len(classes))
        )
    )

    for i, row in enumerate(cm):

        print(
            f"{i:3d}: " +
            " ".join(
                f"{value:4d}"
                for value in row
            )
        )

    # ----------------------------------------
    # Example predictions + confidence
    # ----------------------------------------

    print("\n================================")
    print("SAMPLE PREDICTIONS")
    print("================================")

    for i in range(
        min(20, len(y_test))
    ):

        actual = classes[
            y_test[i]
        ]

        predicted = classes[
            y_pred[i]
        ]

        confidence = probabilities[
            i,
            y_pred[i]
        ]

        print(
            f"{i + 1:02d}. "
            f"Actual: {actual} | "
            f"Predicted: {predicted} | "
            f"Confidence: {confidence:.4f}"
        )

    # ----------------------------------------
    # Save predictions
    # ----------------------------------------

    np.savez(
        os.path.join(
            DATA_DIR,
            "lstm_test_predictions.npz"
        ),
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
        classes=np.asarray(classes)
    )

    print(
        "\nSaved predictions:"
    )

    print(
        "DATA/processed/CICIDS/"
        "lstm_test_predictions.npz"
    )


if __name__ == "__main__":
    main()