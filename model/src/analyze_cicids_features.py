import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\processed\CICIDS"
)

DROP = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label"
]

# Use one representative cleaned file for feature analysis
file = DATA_DIR / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(file, low_memory=False)

features = df.drop(columns=DROP)

# Keep only numeric features
features = features.select_dtypes(include=np.number)

# Correlation matrix
corr = features.corr().abs()

# Find highly correlated feature pairs
pairs = []

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        value = corr.iloc[i, j]

        if value >= 0.95:
            pairs.append(
                (corr.columns[i], corr.columns[j], round(value, 3))
            )

pairs.sort(key=lambda x: x[2], reverse=True)

print("TOTAL NUMERIC FEATURES:", len(features.columns))

print("\nHIGHLY CORRELATED PAIRS (>= 0.95):")

for a, b, value in pairs:
    print(f"{a} <-> {b} : {value}")

print("\nTOTAL HIGH-CORRELATION PAIRS:", len(pairs))