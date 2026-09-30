import csv
import os
from collections import defaultdict

from datetime import datetime


DATA_DIR = "DATA/processed/CICIDS"


for filename in sorted(os.listdir(DATA_DIR)):

    if not filename.endswith(".csv"):
        continue

    path = os.path.join(DATA_DIR, filename)

    attack_times = defaultdict(list)

    with open(path, "r", encoding="latin1", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:

            label = row.get("Label", "").strip()
            timestamp = row.get("Timestamp", "").strip()

            if not timestamp or not label or label == "BENIGN":
                continue

            try:
                timestamp = datetime.strptime(
                    timestamp,
                    "%Y-%m-%d %H:%M:%S"
                )
            except ValueError:
                continue

            attack_times[label].append(timestamp)

    if attack_times:
        print("\n" + filename)

        for label, times in sorted(attack_times.items()):
            print(
                f"  {label}: "
                f"{len(times)} rows | "
                f"{min(times)} -> {max(times)}"
            )