import os
import numpy as np

BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA_DIR = os.path.join(BASE, "DATA", "processed", "CICIDS")

train = np.load(
    os.path.join(DATA_DIR, "train_multiclass.npz")
)

X = train["X"].astype(np.float32)
y = train["y"]

classes = np.load(
    os.path.join(DATA_DIR, "multiclass_classes.npy"),
    allow_pickle=True
)

print("Original:", X.shape)

# Augment classes with fewer than 40 training samples
TARGET = 80
rng = np.random.default_rng(42)

new_X = [X]
new_y = [y]

for cls in classes:

    idx = np.where(y == cls)[0]
    count = len(idx)

    if count == 0 or count >= TARGET:
        continue

    needed = TARGET - count

    print(f"{cls}: {count} -> {TARGET}")

    samples = X[idx]

    generated = []

    for _ in range(needed):

        # Pick two real sequences from the same class
        a = samples[rng.integers(0, count)]
        b = samples[rng.integers(0, count)]

        # Interpolate between them
        alpha = rng.uniform(0.2, 0.8)

        synthetic = (
            alpha * a +
            (1 - alpha) * b
        )

        # Small feature noise
        noise = rng.normal(
            0,
            0.01,
            synthetic.shape
        ).astype(np.float32)

        synthetic = synthetic + noise

        generated.append(synthetic)

    generated = np.asarray(
        generated,
        dtype=np.float32
    )

    new_X.append(generated)

    new_y.append(
        np.array(
            [cls] * needed,
            dtype=y.dtype
        )
    )


X_aug = np.concatenate(new_X, axis=0)
y_aug = np.concatenate(new_y, axis=0)

print("\nAugmented:", X_aug.shape)

print("\nFinal class counts:")

for cls in classes:
    print(
        f"{cls:30s}",
        np.sum(y_aug == cls)
    )


np.savez(
    os.path.join(
        DATA_DIR,
        "train_multiclass_augmented.npz"
    ),
    X=X_aug,
    y=y_aug
)

print(
    "\nSaved:",
    os.path.join(
        DATA_DIR,
        "train_multiclass_augmented.npz"
    )
)