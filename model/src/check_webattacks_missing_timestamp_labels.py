from pathlib import Path
from collections import Counter
import csv

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "DATA/raw/CICID/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

def main():
    print("[START] Check labels on rows with missing timestamps", flush=True)
    total = missing_time = valid_time = 0
    missing_labels = Counter()
    valid_labels = Counter()
    examples = []
    with PATH.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        time_col = next((h for h in headers if h.strip().lower() == "timestamp"), None)
        label_col = next((h for h in headers if h.strip().lower() == "label"), None)
        print(f"[INFO] timestamp={time_col!r}; label={label_col!r}", flush=True)
        if not time_col or not label_col:
            raise RuntimeError(f"Required columns not found. Headers: {headers}")
        for row in reader:
            total += 1
            ts = (row.get(time_col) or "").strip()
            label = (row.get(label_col) or "").strip() or "MISSING_LABEL"
            if ts:
                valid_time += 1
                valid_labels[label] += 1
            else:
                missing_time += 1
                missing_labels[label] += 1
                if len(examples) < 5:
                    examples.append({k: row.get(k) for k in headers[:10]} | {label_col: row.get(label_col), time_col: row.get(time_col)})
            if total % 100000 == 0:
                print(f"[PROGRESS] rows={total:,}", flush=True)
    print(f"[TOTAL] {total:,}; valid timestamp rows={valid_time:,}; missing timestamp rows={missing_time:,}", flush=True)
    print("[MISSING TIMESTAMP LABEL COUNTS]")
    for label, count in missing_labels.most_common():
        print(f"  {label!r}: {count:,}")
    print("[VALID TIMESTAMP LABEL COUNTS]")
    for label, count in valid_labels.most_common():
        print(f"  {label!r}: {count:,}")
    print("[MISSING ROW EXAMPLES]")
    for ex in examples:
        print(ex)
    print("[DONE] Read-only check; no files changed.", flush=True)

if __name__ == "__main__":
    main()
