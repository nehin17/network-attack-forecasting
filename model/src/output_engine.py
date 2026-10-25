def calculate_risk_score(
    future_attack_type,
    future_confidence,
    attack_type,
    attack_confidence
):
    """
    Calculate a 0-100 risk score using the model
    predictions and their confidence.

    Risk is derived from:
    - predicted future attack severity
    - future prediction confidence
    - current attack severity
    - current prediction confidence
    """

    severity_weight = {
        "BENIGN": 0.00,
        "Normal": 0.00,

        "Reconnaissance": 0.35,
        "Analysis": 0.40,

        "Fuzzers": 0.55,
        "Generic": 0.60,

        "PortScan": 0.60,

        "Bot": 0.70,
        "Backdoor": 0.80,
        "Shellcode": 0.85,

        "FTP-Patator": 0.75,
        "SSH-Patator": 0.75,

        "DoS": 0.85,
        "DDoS": 0.95,

        "Exploits": 0.90,
        "Infiltration": 0.95,

        "Heartbleed": 0.90,

        "DoS GoldenEye": 0.90,
        "DoS Hulk": 0.90,
        "DoS Slowhttptest": 0.90,
        "DoS slowloris": 0.90,

        "Web Attack - Brute Force": 0.75,
        "Web Attack - Sql Injection": 0.85,
        "Web Attack - XSS": 0.80,
    }

    future_weight = severity_weight.get(
        future_attack_type,
        0.50
    )

    current_weight = severity_weight.get(
        attack_type,
        0.50
    )

    future_component = (
        future_weight
        * float(future_confidence)
    )

    current_component = (
        current_weight
        * float(attack_confidence)
    )

    # Future prediction is slightly more important
    # because the system is an early-warning system.
    score = (
        0.60 * future_component
        +
        0.40 * current_component
    )

    return round(
        score * 100,
        2
    )


def get_severity(
    risk_score,
    future_attack_type,
    attack_type
):
    """
    Convert risk score + predicted attack type
    into a user-facing severity level.
    """

    attack_types = {
        future_attack_type,
        attack_type
    }

    if attack_types <= {
        "BENIGN",
        "Normal"
    }:
        return "LOW"

    if risk_score >= 80:
        return "CRITICAL"

    if risk_score >= 60:
        return "HIGH"

    if risk_score >= 35:
        return "MEDIUM"

    return "LOW"


def build_output(
    future_attack_type,
    future_confidence,
    attack_type,
    attack_confidence
):
    """
    Build the final user-facing prediction object.
    """

    risk_score = calculate_risk_score(
        future_attack_type,
        future_confidence,
        attack_type,
        attack_confidence
    )

    severity = get_severity(
        risk_score,
        future_attack_type,
        attack_type
    )

    return {
        "future_attack_type": future_attack_type,
        "attack_type": attack_type,
        "future_confidence": round(
            float(future_confidence),
            4
        ),
        "attack_confidence": round(
            float(attack_confidence),
            4
        ),
        "risk_score": risk_score,
        "severity": severity
    }


if __name__ == "__main__":

    result = build_output(
        future_attack_type="PortScan",
        future_confidence=0.91,
        attack_type="Normal",
        attack_confidence=0.94
    )

    print("\nFinal Network Security Output")
    print("=" * 45)

    for key, value in result.items():
        print(
            f"{key}: {value}"
        )