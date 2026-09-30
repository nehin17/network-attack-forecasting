import os
import numpy as np
import xgboost as xgb


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "results",
    "xgboost",
    "xgboost_temporal.json"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "DATA",
    "processed",
    "CICIDS",
    "cicids_selected_features.txt"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "DATA",
    "processed",
    "CICIDS",
    "multiclass_classes.npy"
)


HISTORY_MINUTES = 10
NUM_FEATURES = 70


class FutureAttackPredictor:

    def __init__(self):

        self.features = self._load_features()
        self.classes = np.load(
            CLASS_PATH,
            allow_pickle=True
        )

        self.model = xgb.XGBClassifier()
        self.model.load_model(MODEL_PATH)

    def _load_features(self):

        with open(
            FEATURE_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            features = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if len(features) != NUM_FEATURES:
            raise ValueError(
                f"Expected {NUM_FEATURES} features, "
                f"found {len(features)}"
            )

        return features

    def _create_temporal_features(self, sequence):

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        )

        if sequence.shape != (
            HISTORY_MINUTES,
            NUM_FEATURES
        ):
            raise ValueError(
                "Expected input shape "
                f"({HISTORY_MINUTES}, {NUM_FEATURES}), "
                f"got {sequence.shape}"
            )

        last = sequence[-1]

        mean = np.mean(
            sequence,
            axis=0
        )

        std = np.std(
            sequence,
            axis=0
        )

        delta = (
            sequence[-1]
            - sequence[0]
        )

        temporal_features = np.concatenate(
            [
                last,
                mean,
                std,
                delta
            ]
        )

        return temporal_features.reshape(
            1, -1
        )

    def predict(self, sequence):

        X = self._create_temporal_features(
            sequence
        )

        probabilities = self.model.predict_proba(X)[0]

        prediction_index = int(
            np.argmax(probabilities)
        )

        prediction = self.classes[
            prediction_index
        ]

        confidence = float(
            probabilities[prediction_index]
        )

        return {
            "future_attack_type": str(
                prediction
            ),
            "confidence": confidence
        }


if __name__ == "__main__":

    predictor = FutureAttackPredictor()

    dummy_sequence = np.zeros(
        (10, 70),
        dtype=np.float32
    )

    result = predictor.predict(
        dummy_sequence
    )

    print("\nFuture Attack Prediction")
    print("=" * 40)
    print(
        f"Prediction : "
        f"{result['future_attack_type']}"
    )
    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )