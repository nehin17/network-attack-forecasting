import os
import pandas as pd

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(BASE, "DATA", "processed", "CICIDS")

files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith(".csv")
]

for file in sorted(files):

    path = os.path.join(DATA_DIR, file)

    df = pd.read_csv(path, usecols=["Timestamp", "Label"])

    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    for label in sorted(df["Label"].dropna().unique()):

        rows = df[df["Label"] == label]

        print(
            f"{label:30s} "
            f"rows={len(rows):8d} | "
            f"start={rows['Timestamp'].min()} | "
            f"end={rows['Timestamp'].max()}"
        )