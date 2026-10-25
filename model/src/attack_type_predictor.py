import os
import json
import numpy as np
import xgboost as xgb


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "results",
    "unsw_xgboost",
    "unsw_attack_type_xgboost.json"
)

PREPROCESSING_PATH = os.path.join(
    BASE_DIR,
    "model",
    "results",
    "unsw_xgboost",
    "unsw_preprocessing.json"
)


class AttackTypePredictor:

    def __init__(self):

        with open(
            PREPROCESSING_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            self.preprocessing = json.load(f)

        self.feature_columns = (
            self.preprocessing["feature_columns"]
        )

        self.categorical_columns = (
            self.preprocessing["categorical_columns"]
        )

        self.encoders = (
            self.preprocessing["encoders"]
        )

        self.id_to_class = {
            int(k): v
            for k, v in
            self.preprocessing["id_to_class"].items()
        }

        self.model = xgb.XGBClassifier()

        self.model.load_model(
            MODEL_PATH
        )

    def _prepare_record(self, record):

        values = []

        for feature in self.feature_columns:

            if feature not in record:
                raise ValueError(
                    f"Missing feature: {feature}"
                )

            value = record[feature]

            if feature in self.categorical_columns:

                mapping = self.encoders[feature]

                # Unknown categories → 0
                value = mapping.get(
                    str(value),
                    0
                )

            values.append(value)

        return np.asarray(
            values,
            dtype=np.float32
        ).reshape(1, -1)

    def predict(self, record):

        X = self._prepare_record(record)

        probabilities = (
            self.model.predict_proba(X)[0]
        )

        prediction_id = int(
            np.argmax(probabilities)
        )

        attack_type = self.id_to_class[
            prediction_id
        ]

        confidence = float(
            probabilities[prediction_id]
        )

        return {
            "attack_type": attack_type,
            "confidence": confidence
        }


if __name__ == "__main__":

    import pandas as pd

    test_path = os.path.join(
        BASE_DIR,
        "DATA",
        "processed",
        "UNSW",
        "UNSW_NB15_testing-set.csv"
    )

    df = pd.read_csv(test_path)

    predictor = AttackTypePredictor()

    record = (
        df.iloc[0]
        .drop(
            labels=["attack_cat", "label"],
            errors="ignore"
        )
        .to_dict()
    )

    result = predictor.predict(record)

    print("\nAttack Type Prediction")
    print("=" * 40)
    print(
        f"Actual     : "
        f"{df.iloc[0]['attack_cat']}"
    )
    print(
        f"Predicted  : "
        f"{result['attack_type']}"
    )
    print(
        f"Confidence : "
        f"{result['confidence']:.4f}"
    )