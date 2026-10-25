"""Combine the CICIDS binary future-attack forecaster with the existing UNSW classifier."""
from future_attack_predictor import FutureAttackPredictor
from attack_type_predictor import AttackTypePredictor
from output_engine import build_output


class NetworkPredictionPipeline:
    def __init__(self):
        self.future_predictor = FutureAttackPredictor()
        self.attack_predictor = AttackTypePredictor()

    def predict(self, future_sequence, current_record):
        """
        future_sequence:
            NumPy-compatible matrix with shape (10, N), where N is the feature count
            saved in final_binary_forecaster.joblib. Rows must be consecutive minute
            aggregates and columns must be in the exact saved feature_columns order.

        current_record:
            dictionary with the current-record features expected by AttackTypePredictor
            (the existing UNSW-NB15 classifier input).
        """
        future_result = self.future_predictor.predict(future_sequence)
        attack_result = self.attack_predictor.predict(current_record)
        return build_output(
            future_attack_type=future_result["future_attack_type"],
            future_confidence=future_result["confidence"],
            attack_type=attack_result["attack_type"],
            attack_confidence=attack_result["confidence"],
        )
