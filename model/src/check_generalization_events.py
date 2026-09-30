import csv
import os
from collections import defaultdict

DATA_DIR = "DATA/processed/CICIDS"

ATTACK_LABELS = {
    "BENIGN"
}

def normalize_label(label):
    label = label.strip()

    replacements = {
        "Web Attack Â Brute Force": "Web Attack - Brute Force",
        "Web Attack Â Sql Injection": "Web Attack - Sql Injection",
        "Web Attack Â XSS": "Web Attack - XSS",
        "Web Attack – Brute Force": "Web Attack - Brute Force",
        "Web Attack – Sql Injection": "Web Attack - Sql Injection",
        "Web Attack – XSS": "Web Attack - XSS",
    }

    return replacements.get(label, label)


def get_minutes(path):
    labels_by_minute = {}

    with open(path, "r", encoding="latin1", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            timestamp = row.get("Timestamp", "").strip()
            label = row.get("Label", "").strip()

            if not timestamp or not label:
                continue

            minute = timestamp[:16]

            label = normalize_label(label)

            if minute not in labels_by_minute:
                labels_by_minute[minute] = set()

            if label != "BENIGN":
                labels_by_minute[minute].add(label)

    return labels_by_minute


def count_events(minutes, label):
    times = sorted(
        minute
        for minute, labels in minutes.items()
        if label in labels
    )

    if not times:
        return 0, []

    events = []
    current = [times[0]]

    for previous, current_time in zip(times, times[1:]):

        prev_hour, prev_min = map(int, previous[11:16].split(":"))
        cur_hour, cur_min = map(int, current_time[11:16].split(":"))

        previous_total = prev_hour * 60 + prev_min
        current_total = cur_hour * 60 + cur_min

        if current_total - previous_total <= 1:
            current.append(current_time)
        else:
            events.append(current)
            current = [current_time]

    events.append(current)

    return len(events), events


def main():

    results = defaultdict(list)

    files = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.endswith(".csv")
        and not f.startswith("cicids_")
    )

    for filename in files:

        path = os.path.join(DATA_DIR, filename)

        minutes = get_minutes(path)

        labels = sorted({
            label
            for values in minutes.values()
            for label in values
        })

        print(f"\n{filename}")

        for label in labels:

            event_count, events = count_events(
                minutes,
                label
            )

            if event_count == 0:
                continue

            durations = [
                len(event)
                for event in events
            ]

            print(
                f"  {label}: "
                f"{event_count} event(s), "
                f"durations={durations}"
            )

            results[label].extend(events)

    print("\n" + "=" * 60)
    print("GLOBAL EVENT SUMMARY")
    print("=" * 60)

    for label in sorted(results):

        events = results[label]

        print(
            f"{label}: "
            f"{len(events)} independent event(s)"
        )


if __name__ == "__main__":
    main()