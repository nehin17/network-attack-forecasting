import numpy as np
import json
from collections import Counter

INPUT = "DATA/processed/CICIDS/forecast_windows.npz"
OUTPUT = "DATA/processed/CICIDS"

data = np.load(INPUT, allow_pickle=True)

X = data["X"]
y = data["y"]
features = data["features"]

n = len(X)

# Chronological window split
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]

# Class weights ONLY from training
counts = Counter(y_train)
classes = sorted(counts.keys())
total = len(y_train)

class_weights = {
    cls: total / (len(classes) * counts[cls])
    for cls in classes
}

np.savez_compressed(
    f"{OUTPUT}/train.npz",
    X=X_train,
    y=y_train,
    features=features
)

np.savez_compressed(
    f"{OUTPUT}/val.npz",
    X=X_val,
    y=y_val,
    features=features
)

np.savez_compressed(
    f"{OUTPUT}/test.npz",
    X=X_test,
    y=y_test,
    features=features
)

with open(f"{OUTPUT}/class_weights.json", "w") as f:
    json.dump(class_weights, f, indent=2)

print("TRAIN:", X_train.shape, y_train.shape)
print("VAL:  ", X_val.shape, y_val.shape)
print("TEST: ", X_test.shape, y_test.shape)

print("\nTraining distribution:")
for cls, count in counts.items():
    print(f"{cls}: {count}")

print("\nValidation distribution:")
print(dict(Counter(y_val)))

print("\nTest distribution:")
print(dict(Counter(y_test)))

print("\nClass weights:")
for cls, weight in class_weights.items():
    print(f"{cls}: {weight:.4f}")

print("\nSaved.")