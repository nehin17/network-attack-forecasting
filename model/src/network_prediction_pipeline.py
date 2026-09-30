from future_attack_predictor import FutureAttackPredictor
from attack_type_predictor import AttackTypePredictor
from output_engine import build_output


class NetworkPredictionPipeline:

    def __init__(self):
        self.future_predictor = FutureAttackPredictor()
        self.attack_predictor = AttackTypePredictor()

    def predict(
        self,
        future_sequence,
        current_record
    ):
        """
        Run the complete ML prediction pipeline.

        future_sequence:
            numpy array with shape (10, 70)

        current_record:
            dictionary containing the 42 UNSW
            model input features.
        """

        future_result = (
            self.future_predictor.predict(
                future_sequence
            )
        )

        attack_result = (
            self.attack_predictor.predict(
                current_record
            )
        )

        final_result = build_output(
            future_attack_type=(
                future_result[
                    "future_attack_type"
                ]
            ),
            future_confidence=(
                future_result[
                    "confidence"
                ]
            ),
            attack_type=(
                attack_result[
                    "attack_type"
                ]
            ),
            attack_confidence=(
                attack_result[
                    "confidence"
                ]
            )
        )

        return final_result


if __name__ == "__main__":

    import os
    import numpy as np
    import pandas as pd

    BASE_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )

    # Real CICIDS sequence
    cicids_path = os.path.join(
        BASE_DIR,
        "DATA",
        "processed",
        "CICIDS",
        "test_multiclass.npz"
    )

    cicids = np.load(
        cicids_path,
        allow_pickle=True
    )

    future_sequence = cicids["X"][0]

    # Real UNSW record
    unsw_path = os.path.join(
        BASE_DIR,
        "DATA",
        "processed",
        "UNSW",
        "UNSW_NB15_testing-set.csv"
    )

    unsw = pd.read_csv(
        unsw_path
    )

    current_record = (
        unsw.iloc[0]
        .drop(
            labels=[
                "attack_cat",
                "label"
            ],
            errors="ignore"
        )
        .to_dict()
    )

    pipeline = NetworkPredictionPipeline()

    result = pipeline.predict(
        future_sequence,
        current_record
    )

    print("\nFINAL NETWORK PREDICTION")
    print("=" * 50)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )