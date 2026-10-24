
from pathlib import Path
import pandas as pd

DATA_PATH = Path("DATA/raw/TON_IoT/Network/train_test_network.csv")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

print("TON_IoT NETWORK DATASET PROFILE")
print("=" * 55)
print(f"File size: {DATA_PATH.stat().st_size / (1024**2):.2f} MB")

df = pd.read_csv(DATA_PATH, low_memory=False)

print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumns and data types:")
print(df.dtypes.to_string())

print("\nMissing values (top 20):")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(20).to_string())

print("\nPotential label / timestamp columns:")
for col in df.columns:
    if any(word in col.lower() for word in
           ["label", "attack", "type", "time", "date"]):
        print(f"\n{col}:")
        print(df[col].value_counts(dropna=False).head(20).to_string())

print("\nNumeric feature summary:")
print(df.select_dtypes(include="number").describe().round(3).to_string())

print("\nDuplicate rows:", int(df.duplicated().sum()))
print("\nPROFILE COMPLETE")
