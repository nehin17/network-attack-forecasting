
from output_engine import build_output


class NetworkPredictionPipeline:
    """Combine existing current and future predictors into four outputs."""

    def __init__(self, future_predictor, attack_predictor):
        self.future_predictor = future_predictor
        self.attack_predictor = attack_predictor

    def predict(self, future_sequence, current_record):
        # Current detection: existing multiclass classifier.
        current = self.attack_predictor.predict(current_record)

        # Future forecasting: existing binary temporal model.
        future = self.future_predictor.predict(future_sequence)

        # A binary ATTACK forecast does not identify the future attack type.
        # Use its probability as the forecast risk component, while retaining
        # the current classifier's actual predicted attack category.
        result = build_output(
            future_attack_type=future["future_attack_type"],
            future_confidence=future["confidence"],
            attack_type=current["attack_type"],
            attack_confidence=current["confidence"],
        )

        return {
            "current_detection": {
                "attack_type": current["attack_type"],
                "confidence": current["confidence"],
            },
            "future_forecast": {
                "prediction": future["future_attack_type"],
                "attack_probability": future["attack_probability"],
                "threshold": future["threshold"],
                "model": future["model"],
            },
            "severity": result["severity"],
            "risk_score": result["risk_score"],
        }

from output_engine import build_output


class NetworkPredictionPipeline:
    """Combine existing current and future predictors into four outputs."""

    def __init__(self, future_predictor, attack_predictor):
        self.future_predictor = future_predictor
        self.attack_predictor = attack_predictor

    def predict(self, future_sequence, current_record):
        # Current detection: existing multiclass classifier.
        current = self.attack_predictor.predict(current_record)

        # Future forecasting: existing binary temporal model.
        future = self.future_predictor.predict(future_sequence)

        # A binary ATTACK forecast does not identify the future attack type.
        # Use its probability as the forecast risk component, while retaining
        # the current classifier's actual predicted attack category.
        result = build_output(
            future_attack_type=future["future_attack_type"],
            future_confidence=future["confidence"],
            attack_type=current["attack_type"],
            attack_confidence=current["confidence"],
        )

        return {
            "current_detection": {
                "attack_type": current["attack_type"],
                "confidence": current["confidence"],
            },
            "future_forecast": {
                "prediction": future["future_attack_type"],
                "attack_probability": future["attack_probability"],
                "threshold": future["threshold"],
                "model": future["model"],
            },
            "severity": result["severity"],
            "risk_score": result["risk_score"],
        }
