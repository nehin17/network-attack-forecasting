import os
from collections import Counter
import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE, "DATA", "processed", "CICIDS")
SPLITS = ("train", "val", "test", "coverage_test")
CLASSES_PATH = os.path.join(DATA_DIR, "multiclass_classes.npy")

classes = np.load(CLASSES_PATH, allow_pickle=True).tolist()
print("CICIDS TEMPORAL DATASET AUDIT")
print("=" * 72)
print(f"Data directory: {DATA_DIR}")
print(f"Classes ({len(classes)}): {classes}")

all_counts = Counter()
loaded = {}

for split in SPLITS:
    path = os.path.join(DATA_DIR, f"{split}_multiclass.npz")
    if not os.path.exists(path):
        print(f"\n{split}: FILE NOT FOUND ({path})")
        continue

    with np.load(path, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"].astype(str)

    loaded[split] = (X, y)
    counts = Counter(y.tolist())
    all_counts.update(counts)

    print(f"\n{split.upper()}: {len(y)} windows; X shape={X.shape}")
    for label in classes:
        n = counts.get(str(label), 0)
        if n:
            print(f"  {str(label):30s} {n:6d} ({n / len(y):6.1%})")
    unknown = sorted(set(counts) - set(map(str, classes)))
    if unknown:
        print(f"  Labels not in class list: {unknown}")

print("\n" + "=" * 72)
print("CLASS COVERAGE BY SPLIT")
print(f"{'Class':30s} {'train':>8s} {'val':>8s} {'test':>8s} {'coverage':>10s}")
for label in classes:
    row = [str(label)]
    for split in ("train", "val", "test", "coverage_test"):
        if split in loaded:
            row.append(str(Counter(loaded[split][1].tolist()).get(str(label), 0)))
        else:
            row.append("MISSING")
    print(f"{row[0]:30s} {row[1]:>8s} {row[2]:>8s} {row[3]:>8s} {row[4]:>10s}")

print("\nRECOMMENDATION")
test_counts = Counter(loaded.get("test", (None, np.array([], dtype=str)))[1].tolist())
test_present = [c for c in classes if test_counts.get(str(c), 0) > 0]
test_missing = [c for c in classes if test_counts.get(str(c), 0) == 0]
print(f"Chronological test has {len(test_present)}/{len(classes)} classes.")
print(f"Classes absent from chronological test: {test_missing}")
print("This script is read-only: it does not modify or overwrite any dataset or model.")
print("Use this report to choose a defensible contiguous time-block test split; do not randomly")
print("shuffle overlapping temporal windows just to force all classes into the test set.")
