"""Binary next-minute CICIDS2017 attack forecaster.

Input contract:
    sequence: array-like, shape (history_minutes, number_of_saved_feature_columns)
    Each row is one consecutive minute of aggregate numeric CICIDS flow features,
    including flow_count, in the exact feature_columns order stored in the bundle.
"""
from pathlib import Path
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "model" / "artifacts" / "cicids_binary_forecast" / "final_binary_forecaster.joblib"


class FutureAttackPredictor:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Binary forecaster bundle not found: {model_path}. "
                "Run model/src/finalize_cicids_binary_forecaster.py first."
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.feature_columns = list(bundle["feature_columns"])
        self.history_minutes = int(bundle.get("history_minutes", 10))
        self.selected_model = bundle.get("selected_model", "xgboost")

    def _engineer(self, sequence):
        seq = np.asarray(sequence, dtype=np.float32)
        expected = (self.history_minutes, len(self.feature_columns))
        if seq.shape != expected:
            raise ValueError(
                f"Expected sequence shape {expected} (history minutes x saved feature count), "
                f"received {seq.shape}. The old 10x70 multiclass sequence is not compatible "
                "with this binary model; supply per-minute aggregates in the saved schema."
            )
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        engineered = np.concatenate([
            seq[-1],
            seq.mean(axis=0),
            seq.std(axis=0),
            seq[-1] - seq[0],
        ])
        return engineered.reshape(1, -1)

    def predict(self, sequence):
        X = self._engineer(sequence)
        attack_probability = float(self.model.predict_proba(X)[0, 1])
        is_attack = attack_probability >= self.threshold
        return {
            "future_attack_type": "ATTACK" if is_attack else "BENIGN",
            "confidence": attack_probability if is_attack else 1.0 - attack_probability,
            "attack_probability": attack_probability,
            "threshold": self.threshold,
            "model": self.selected_model,
        }
