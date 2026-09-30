import csv
import os
from collections import defaultdict, Counter
from datetime import datetime

import numpy as np


RAW_DIR = "DATA/processed/CICIDS"
FEATURE_FILE = "DATA/processed/CICIDS/cicids_selected_features.txt"
OUTPUT_DIR = "DATA/processed/CICIDS"

HISTORY_MINUTES = 10
HORIZON_MINUTES = 1

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def load_selected_features():
    with open(FEATURE_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def parse_time(value):
    return datetime.strptime(
        value.strip(),
        "%Y-%m-%d %H:%M:%S"
    )


def normalize_label(label):

    label = label.strip()

    replacements = {
        "Web Attack Â Brute Force": "Web Attack - Brute Force",
        "Web Attack Â Sql Injection": "Web Attack - Sql Injection",
        "Web Attack Â XSS": "Web Attack - XSS",

        "Web Attack Â\x96 Brute Force":
            "Web Attack - Brute Force",

        "Web Attack Â\x96 Sql Injection":
            "Web Attack - Sql Injection",

        "Web Attack Â\x96 XSS":
            "Web Attack - XSS",

        "Web Attack – Brute Force":
            "Web Attack - Brute Force",

        "Web Attack – Sql Injection":
            "Web Attack - Sql Injection",

        "Web Attack – XSS":
            "Web Attack - XSS",
    }

    return replacements.get(label, label)


def process_file(path, selected_features):

    filename = os.path.basename(path)

    print(f"\nProcessing: {filename}")

    minute_values = defaultdict(
        lambda: defaultdict(list)
    )

    minute_attack_labels = defaultdict(list)

    with open(
        path,
        "r",
        encoding="latin1",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            timestamp = row.get("Timestamp", "")
            label = row.get("Label", "")

            if not timestamp or not label:
                continue

            try:
                ts = parse_time(timestamp)
            except ValueError:
                continue

            minute = ts.replace(second=0)

            # -----------------------------
            # Feature aggregation
            # -----------------------------

            for feature in selected_features:

                try:

                    value = float(
                        row.get(feature, "")
                    )

                    if np.isfinite(value):

                        minute_values[
                            minute
                        ][feature].append(value)

                except (ValueError, TypeError):
                    continue

            # -----------------------------
            # Attack labels
            # -----------------------------

            clean_label = normalize_label(label)

            if clean_label != "BENIGN":

                minute_attack_labels[
                    minute
                ].append(clean_label)

    timestamps = sorted(
        minute_values.keys()
    )

    if len(timestamps) < (
        HISTORY_MINUTES +
        HORIZON_MINUTES
    ):
        return [], [], []

    X_minutes = []
    minute_labels = []

    for timestamp in timestamps:

        # -----------------------------
        # One feature vector per minute
        # -----------------------------

        vector = []

        for feature in selected_features:

            values = minute_values[
                timestamp
            ].get(feature, [])

            if values:

                vector.append(
                    float(np.mean(values))
                )

            else:

                vector.append(0.0)

        X_minutes.append(vector)

        # -----------------------------
        # One attack label per minute
        # -----------------------------

        attacks = minute_attack_labels.get(
            timestamp,
            []
        )

        if attacks:

            minute_label = Counter(
                attacks
            ).most_common(1)[0][0]

        else:

            minute_label = "BENIGN"

        minute_labels.append(
            minute_label
        )

    X_minutes = np.asarray(
        X_minutes,
        dtype=np.float32
    )

    X_minutes = np.nan_to_num(
        X_minutes,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return (
        timestamps,
        X_minutes,
        minute_labels
    )


def build_windows(
    X_minutes,
    labels,
    timestamps,
    start,
    end
):

    split_X = X_minutes[start:end]
    split_labels = labels[start:end]
    split_timestamps = timestamps[start:end]

    samples = []
    targets = []
    target_times = []

    n = len(split_X)

    for i in range(
        HISTORY_MINUTES,
        n - HORIZON_MINUTES + 1
    ):

        history = split_X[
            i - HISTORY_MINUTES:i
        ]

        future_labels = split_labels[
            i:i + HORIZON_MINUTES
        ]

        target = "BENIGN"

        for future_label in future_labels:

            if future_label != "BENIGN":

                target = future_label
                break

        samples.append(history)
        targets.append(target)

        target_times.append(
            split_timestamps[i]
        )

    return samples, targets, target_times


def main():

    selected_features = (
        load_selected_features()
    )

    files = sorted(
        f
        for f in os.listdir(RAW_DIR)
        if f.endswith(".csv")
    )

    all_train_X = []
    all_train_y = []

    all_val_X = []
    all_val_y = []

    all_test_X = []
    all_test_y = []

    # Extra real-data coverage set
    coverage_X = []
    coverage_y = []

    for filename in files:

        path = os.path.join(
            RAW_DIR,
            filename
        )

        timestamps, X, labels = process_file(
            path,
            selected_features
        )

        if not timestamps:
            continue

        n = len(timestamps)

        # --------------------------------------
        # Original chronological split
        # --------------------------------------

        train_end = int(
            n * TRAIN_RATIO
        )

        val_end = int(
            n * (TRAIN_RATIO + VAL_RATIO)
        )

        train_X, train_y, _ = build_windows(
            X,
            labels,
            timestamps,
            0,
            train_end
        )

        val_X, val_y, _ = build_windows(
            X,
            labels,
            timestamps,
            train_end,
            val_end
        )

        test_X, test_y, _ = build_windows(
            X,
            labels,
            timestamps,
            val_end,
            n
        )

        all_train_X.extend(train_X)
        all_train_y.extend(train_y)

        all_val_X.extend(val_X)
        all_val_y.extend(val_y)

        all_test_X.extend(test_X)
        all_test_y.extend(test_y)

        # --------------------------------------
        # REAL CLASS COVERAGE WINDOWS
        #
        # Built from the complete real capture.
        # One representative window per class
        # occurrence is collected.
        # --------------------------------------

        full_X, full_y, full_times = build_windows(
            X,
            labels,
            timestamps,
            0,
            n
        )

        seen_class = set()

        for sample, target, target_time in zip(
            full_X,
            full_y,
            full_times
        ):

            if target not in seen_class:

                coverage_X.append(sample)
                coverage_y.append(target)

                seen_class.add(target)

        print(
            f"  minutes: {n} | "
            f"train: {len(train_y)} | "
            f"val: {len(val_y)} | "
            f"test: {len(test_y)}"
        )

    # --------------------------------------
    # Convert arrays
    # --------------------------------------

    X_train = np.asarray(
        all_train_X,
        dtype=np.float32
    )

    y_train = np.asarray(
        all_train_y,
        dtype="<U64"
    )

    X_val = np.asarray(
        all_val_X,
        dtype=np.float32
    )

    y_val = np.asarray(
        all_val_y,
        dtype="<U64"
    )

    X_test = np.asarray(
        all_test_X,
        dtype=np.float32
    )

    y_test = np.asarray(
        all_test_y,
        dtype="<U64"
    )

    X_coverage = np.asarray(
        coverage_X,
        dtype=np.float32
    )

    y_coverage = np.asarray(
        coverage_y,
        dtype="<U64"
    )

    # --------------------------------------
    # Classes
    # --------------------------------------

    all_classes = sorted(
        set(y_train.tolist())
        | set(y_val.tolist())
        | set(y_test.tolist())
        | set(y_coverage.tolist())
    )

    # --------------------------------------
    # Save original chronological datasets
    # --------------------------------------

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "train_multiclass.npz"
        ),
        X=X_train,
        y=y_train
    )

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "val_multiclass.npz"
        ),
        X=X_val,
        y=y_val
    )

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "test_multiclass.npz"
        ),
        X=X_test,
        y=y_test
    )

    # --------------------------------------
    # Save real 15-class coverage set
    # --------------------------------------

    np.savez(
        os.path.join(
            OUTPUT_DIR,
            "coverage_test_multiclass.npz"
        ),
        X=X_coverage,
        y=y_coverage
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "multiclass_classes.npy"
        ),
        np.asarray(
            all_classes,
            dtype="<U64"
        )
    )

    # --------------------------------------
    # Report
    # --------------------------------------

    print(
        "\n=========================================="
    )

    print(
        "FINAL MULTICLASS FORECASTING DATA"
    )

    print(
        "=========================================="
    )

    print(
        "History:",
        HISTORY_MINUTES,
        "minutes"
    )

    print(
        "Forecast horizon:",
        HORIZON_MINUTES,
        "minute"
    )

    print(
        "Features:",
        X_train.shape[2]
    )

    print(
        "\nTRAIN:",
        X_train.shape
    )

    print(
        "VAL:  ",
        X_val.shape
    )

    print(
        "TEST: ",
        X_test.shape
    )

    print(
        "COVERAGE TEST:",
        X_coverage.shape
    )

    print(
        "\nTRAIN classes:"
    )
    print(
        Counter(y_train)
    )

    print(
        "\nVALIDATION classes:"
    )
    print(
        Counter(y_val)
    )

    print(
        "\nTEST classes:"
    )
    print(
        Counter(y_test)
    )

    print(
        "\nCOVERAGE TEST classes:"
    )
    print(
        Counter(y_coverage)
    )

    print(
        "\nALL CLASSES:"
    )

    for label in all_classes:
        print(
            f"  {label}"
        )

    print(
        "\n=========================================="
    )

    if len(all_classes) == 15:

        print(
            "DATASET STATUS: 15 CLASSES PRESENT"
        )

        print(
            "Original chronological test preserved."
        )

        print(
            "Real 15-class coverage test created."
        )

    else:

        print(
            "DATASET STATUS: CHECK REQUIRED"
        )

    print(
        "=========================================="
    )


if __name__ == "__main__":
    main()