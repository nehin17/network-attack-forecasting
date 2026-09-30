import pandas as pd
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\processed\UNSW"
)

for file in DATA_DIR.glob("*.csv"):

    df = pd.read_csv(file, low_memory=False)

    print(f"\n{file.name}")
    print("Rows:", len(df))
    print("attack_cat missing:", df["attack_cat"].isna().sum())
    print("label missing:", df["label"].isna().sum())
    print("Attack categories:")
    print(df["attack_cat"].value_counts(dropna=False))
    print("Binary labels:")
    print(df["label"].value_counts(dropna=False))