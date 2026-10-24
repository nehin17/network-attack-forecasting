
from pathlib import Path
import pandas as pd

path = Path("DATA/raw/TON_IoT/GroundTruth/GroundTruth_Network_1.csv")

if not path.exists():
    raise FileNotFoundError(f"File not found: {path.resolve()}")

print("File size MB:", round(path.stat().st_size / 1e6, 2))

# Inspect the schema and a small sample first.
sample = pd.read_csv(path, nrows=5, low_memory=False)
print("\nCOLUMNS:")
print(sample.columns.tolist())
print("\nSAMPLE:")
print(sample.to_string(index=False))

# Count rows and collect value counts for likely event fields.
counts = {}
rows = 0
for chunk in pd.read_csv(path, chunksize=100_000, low_memory=False):
    rows += len(chunk)
    for col in chunk.columns:
        if any(k in col.lower() for k in
               ["attack", "type", "label", "time", "date"]):
            vals = chunk[col].astype(str).value_counts()
            counts[col] = counts.get(col, pd.Series(dtype="int64")).add(
                vals, fill_value=0
            )

print(f"\nTOTAL ROWS: {rows:,}")
for col, values in counts.items():
    print(f"\nVALUE COUNTS — {col}:")
    print(values.sort_values(ascending=False).head(25).to_string())
