import csv
import os
import numpy as np

INPUT = "DATA/processed/CICIDS/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
OUTPUT = "DATA/processed/CICIDS/cicids_selected_features.txt"

DROP = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"}

with open(INPUT, "r", encoding="latin1", newline="") as f:
    reader = csv.DictReader(f)
    features = [c for c in reader.fieldnames if c not in DROP]

    sums = np.zeros(len(features))
    sums_sq = np.zeros(len(features))
    count = 0

    for row in reader:
        values = []
        valid = True

        for feature in features:
            try:
                value = float(row[feature])
                if not np.isfinite(value):
                    valid = False
                    break
                values.append(value)
            except (ValueError, TypeError):
                valid = False
                break

        if valid:
            x = np.asarray(values, dtype=np.float64)
            sums += x
            sums_sq += x * x
            count += 1

        if count >= 100000:
            break

variance = (sums_sq / count) - (sums / count) ** 2
selected = [f for f, v in zip(features, variance) if v > 0]

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    f.write("\n".join(selected))

print(f"Rows analysed: {count}")
print(f"Candidate features: {len(features)}")
print(f"Selected features: {len(selected)}")
print(f"Saved to: {OUTPUT}")
