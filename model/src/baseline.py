import numpy as np
from collections import Counter

DATA = "DATA/processed/CICIDS"

train = np.load(f"{DATA}/train.npz", allow_pickle=True)
test = np.load(f"{DATA}/test.npz", allow_pickle=True)

y_train = train["y"]
y_test = test["y"]

majority_class = Counter(y_train).most_common(1)[0][0]

predictions = np.full(len(y_test), majority_class)

accuracy = np.mean(predictions == y_test)

print("Baseline prediction:", majority_class)
print("Test samples:", len(y_test))
print("Correct:", np.sum(predictions == y_test))
print("Accuracy:", round(accuracy, 4))