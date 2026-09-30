import pandas as pd
from pathlib import Path

DATA_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\processed\CICIDS"
)

for file in DATA_DIR.glob("*.csv"):
    df = pd.read_csv(file, low_memory=False)

    print(f"\n{file.name}")
    print("Rows:", len(df))
    print("Timestamp missing:", df["Timestamp"].isna().sum())
    print("Label missing:", df["Label"].isna().sum())
    print("Unique labels:", df["Label"].nunique())
    print("Labels:", df["Label"].unique())