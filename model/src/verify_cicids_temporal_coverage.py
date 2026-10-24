from pathlib import Path
from collections import Counter
import csv, time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "DATA" / "raw" / "CICID"
OUT_DIR = ROOT / "model" / "results" / "dataset_audit"
TXT = OUT_DIR / "cicids_temporal_coverage.txt"
CSV_OUT = OUT_DIR / "cicids_temporal_coverage.csv"
FILES = [
"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
"Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
"Friday-WorkingHours-Morning.pcap_ISCX.csv",
"Monday-WorkingHours.pcap_ISCX.csv",
"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
"Tuesday-WorkingHours.pcap_ISCX.csv",
"Wednesday-workingHours.pcap_ISCX.csv",
]

def norm(s):
    return "".join(ch.lower() for ch in str(s or "").strip().lstrip("\ufeff") if ch.isalnum())

def parse_time(s):
    s = str(s or "").strip()
    for fmt in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = [
        "CICIDS2017 FULL-CAPTURE TEMPORAL VERIFICATION",
        "Read-only full-file scan. No training, resampling, or model/data changes.",
        "Each capture is analyzed separately; no raw/processed duplicates are included."
    ]
    summaries = []
    print("[START] Full-file CICIDS scan (read-only)", flush=True)
    print(f"[ROOT] {ROOT}", flush=True)
    for idx, name in enumerate(FILES, 1):
        path = DATA_DIR / name
        print(f"[FILE {idx}/{len(FILES)}] {name}", flush=True)
        entry = {"file": name, "status": "OK", "rows": 0, "timestamp_column": "",
                 "label_column": "", "valid_timestamps": 0, "invalid_timestamps": 0,
                 "out_of_order_rows": 0, "classes": 0, "benign_rows": 0,
                 "nonbenign_rows": 0, "minute_bins": 0, "ten_minute_bins": 0,
                 "ten_minute_bins_with_attack": 0, "dominant_label_changes": 0,
                 "first_timestamp": "", "last_timestamp": "", "elapsed_seconds": 0}
        if not path.exists():
            entry["status"] = "MISSING"
            report.extend(["", f"FILE: {name}", "STATUS: MISSING"])
            summaries.append(entry)
            print("[SKIP] File missing", flush=True)
            continue
        start = time.time()
        labels = Counter()
        minute_bins = Counter()
        ten_bins = {}
        prev = None
        out_of_order = valid = invalid = rows = benign = nonbenign = 0
        min_dt = max_dt = None
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                by_norm = {norm(h): h for h in headers}
                time_col = by_norm.get("timestamp")
                label_col = by_norm.get("label")
                entry["timestamp_column"] = time_col or ""
                entry["label_column"] = label_col or ""
                if not time_col or not label_col:
                    entry["status"] = "MISSING_REQUIRED_COLUMN"
                    report.extend(["", f"FILE: {name}", f"Headers: {headers}",
                                   f"Timestamp column: {time_col!r}", f"Label column: {label_col!r}"])
                    summaries.append(entry)
                    continue
                for r in reader:
                    rows += 1
                    label = str(r.get(label_col, "") or "").strip() or "MISSING_LABEL"
                    labels[label] += 1
                    if label.upper() == "BENIGN":
                        benign += 1
                    else:
                        nonbenign += 1
                    dt = parse_time(r.get(time_col, ""))
                    if dt is None:
                        invalid += 1
                        continue
                    valid += 1
                    if prev is not None and dt < prev:
                        out_of_order += 1
                    prev = dt
                    min_dt = dt if min_dt is None or dt < min_dt else min_dt
                    max_dt = dt if max_dt is None or dt > max_dt else max_dt
                    minute = dt.replace(second=0, microsecond=0)
                    ten = dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)
                    minute_bins[minute] += 1
                    ten_bins.setdefault(ten, Counter())[label] += 1
        except Exception as e:
            entry["status"] = f"ERROR: {type(e).__name__}: {e}"
            report.extend(["", f"FILE: {name}", entry["status"]])
            summaries.append(entry)
            print(f"[ERROR] {e}", flush=True)
            continue
        dominant = []
        attack_bins = 0
        for _, counts in sorted(ten_bins.items()):
            if any(k.upper() != "BENIGN" for k in counts):
                attack_bins += 1
            top = max(counts.values())
            dominant.append(sorted(k for k, v in counts.items() if v == top)[0])
        changes = sum(a != b for a, b in zip(dominant, dominant[1:]))
        entry.update({
            "rows": rows, "valid_timestamps": valid, "invalid_timestamps": invalid,
            "out_of_order_rows": out_of_order, "classes": len(labels),
            "benign_rows": benign, "nonbenign_rows": nonbenign,
            "minute_bins": len(minute_bins), "ten_minute_bins": len(ten_bins),
            "ten_minute_bins_with_attack": attack_bins, "dominant_label_changes": changes,
            "first_timestamp": min_dt.isoformat(sep=" ") if min_dt else "",
            "last_timestamp": max_dt.isoformat(sep=" ") if max_dt else "",
            "elapsed_seconds": round(time.time() - start, 2)
        })
        report.extend([
            "", "=" * 76, f"FILE: {name}", f"Rows scanned: {rows:,}",
            f"Timestamp column: {time_col!r}", f"Label column: {label_col!r}",
            f"Valid timestamps: {valid:,}/{rows:,} ({valid / rows:.2%})" if rows else "No rows",
            f"Invalid timestamps: {invalid:,}", f"Out-of-order timestamp rows: {out_of_order:,}",
            f"Timestamp range: {entry['first_timestamp']} -> {entry['last_timestamp']}",
            f"BENIGN rows: {benign:,}", f"Non-BENIGN rows: {nonbenign:,}",
            f"Unique labels ({len(labels)}):"
        ])
        report.extend([f"  {k}: {v:,}" for k, v in sorted(labels.items())])
        report.extend([
            f"Occupied minute bins: {len(minute_bins):,}",
            f"Occupied 10-minute bins: {len(ten_bins):,}",
            f"10-minute bins containing any non-BENIGN row: {attack_bins:,}",
            f"Changes in dominant label across consecutive occupied 10-minute bins: {changes:,}",
            f"Elapsed seconds: {entry['elapsed_seconds']}"
        ])
        summaries.append(entry)
        print(f"[OK] rows={rows:,}; classes={len(labels)}; valid timestamps={valid:,}; "
              f"attack-containing 10m bins={attack_bins:,}; elapsed={entry['elapsed_seconds']}s", flush=True)
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    TXT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[DONE] Text report: {TXT}", flush=True)
    print(f"[DONE] CSV summary: {CSV_OUT}", flush=True)

if __name__ == "__main__":
    main()
