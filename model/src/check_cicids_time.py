import csv
from datetime import datetime

INPUT = "DATA/processed/CICIDS/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

timestamps = []

with open(INPUT, "r", encoding="latin1", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        try:
            timestamps.append(
                datetime.strptime(row["Timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
            )
        except (ValueError, TypeError):
            pass

print("Rows checked:", len(timestamps))
print("First timestamp:", min(timestamps))
print("Last timestamp:", max(timestamps))

is_sorted = all(
    timestamps[i] <= timestamps[i + 1]
    for i in range(len(timestamps) - 1)
)

print("Chronologically sorted:", is_sorted)