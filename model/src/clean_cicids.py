import pandas as pd
import numpy as np
from pathlib import Path

RAW_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\raw\CICID"
)

OUT_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\processed\CICIDS"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


for file in RAW_DIR.glob("*.csv"):

    print(f"\nProcessing: {file.name}")

    df = pd.read_csv(
        file,
        encoding="latin1",
        low_memory=False
    )

    original_rows = len(df)

    # 1. Clean column names
    df.columns = df.columns.str.strip()

    # 2. Remove completely empty rows
    df = df.dropna(how="all")

    # 3. Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4. Convert timestamp
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce"
    )

    # 5. Remove rows without timestamp or label
    df = df.dropna(subset=["Timestamp", "Label"])

    # 6. Convert feature columns to numeric
    for col in df.columns:
        if col not in ["Timestamp", "Label", "Flow ID", "Source IP", "Destination IP"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    output_file = OUT_DIR / file.name

    df.to_csv(output_file, index=False)

    print(f"Original rows : {original_rows}")
    print(f"Cleaned rows  : {len(df)}")
    print(f"Removed rows  : {original_rows - len(df)}")
    print(f"Saved to      : {output_file}")