import pandas as pd
from pathlib import Path

RAW_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\raw\UNSWNB"
)

OUT_DIR = Path(
    r"C:\Users\Neha\Desktop\semester 07\minor project (4)"
    r"\network-attack-forecasting\DATA\processed\UNSW"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

files = [
    "UNSW_NB15_training-set.csv",
    "UNSW_NB15_testing-set.csv"
]

for filename in files:

    print(f"\nProcessing: {filename}")

    file = RAW_DIR / filename

    df = pd.read_csv(file, low_memory=False)

    original_rows = len(df)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    output_file = OUT_DIR / filename
    df.to_csv(output_file, index=False)

    print(f"Original rows : {original_rows}")
    print(f"Cleaned rows  : {len(df)}")
    print(f"Removed rows  : {original_rows - len(df)}")
    print(f"Columns       : {len(df.columns)}")
    print(f"Saved to      : {output_file}")