"""
TARGETED TEMPORAL SUITABILITY CHECK — no model training, no dataset modification.
Run from project root:
  .\.venv\Scripts\python.exe -u model\src\check_temporal_suitability.py

Reads only a bounded sample from each unique candidate source file and writes:
  model\results\dataset_audit\temporal_suitability_report.txt
"""
from pathlib import Path
from collections import Counter
import csv
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "model" / "results" / "dataset_audit"
OUT_FILE = OUT_DIR / "temporal_suitability_report.txt"
MAX_ROWS = 30000

FILES = [
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Friday-WorkingHours-Morning.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Monday-WorkingHours.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Tuesday-WorkingHours.pcap_ISCX.csv"),
    ("CICIDS2017", ROOT / "DATA/raw/CICID/Wednesday-workingHours.pcap_ISCX.csv"),
    ("UNSW-NB15", ROOT / "DATA/processed/UNSW/UNSW_NB15_training-set.csv"),
    ("UNSW-NB15", ROOT / "DATA/processed/UNSW/UNSW_NB15_testing-set.csv"),
    ("TON_IoT", ROOT / "DATA/raw/TON_IoT/Network/train_test_network.csv"),
]

def norm(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

def find_col(columns, preferred, hints):
    by_norm = {norm(c): c for c in columns}
    for name in preferred:
        if norm(name) in by_norm:
            return by_norm[norm(name)]
    matches = [c for c in columns if any(h in norm(c) for h in hints)]
    return matches[0] if matches else None

def inspect_file(dataset, path):
    result = [f"\n{'='*78}", f"DATASET: {dataset}", f"FILE: {path.relative_to(ROOT)}"]
    if not path.is_file():
        result.append("STATUS: FILE NOT FOUND")
        print(f"[SKIP] Missing: {path}", flush=True)
        return result

    try:
        import pandas as pd
    except ImportError:
        result.append("ERROR: pandas is unavailable in this environment.")
        return result

    try:
        df = pd.read_csv(path, nrows=MAX_ROWS, low_memory=False, encoding_errors="replace")
    except TypeError:
        df = pd.read_csv(path, nrows=MAX_ROWS, low_memory=False, encoding="latin1")
    except Exception as exc:
        result.append(f"ERROR reading file: {type(exc).__name__}: {exc}")
        print(f"[ERROR] {path.name}: {exc}", flush=True)
        return result

    cols = list(df.columns)
    time_col = find_col(cols, ["Timestamp", "stime", "StartTime", "ts", "time", "datetime"], ["timestamp", "datetime", "starttime", "stime"])
    label_col = find_col(cols, ["Label", "attack_cat", "type", "attack_type", "label"], ["attackcat", "attacktype", "label", "class", "category"])
    # TON_IoT commonly has a binary label and a type/category. Prefer the multiclass type if present.
    if dataset == "TON_IoT":
        label_col = find_col(cols, ["type", "attack_type", "attack_cat", "label"], ["attacktype", "attackcat", "type", "label"])
    result.append(f"Rows sampled: {len(df):,} (cap {MAX_ROWS:,}; not a full-file count)")
    result.append(f"Columns ({len(cols)}): {', '.join(map(str, cols))}")
    result.append(f"Chosen timestamp column: {time_col!r}")
    result.append(f"Chosen attack target candidate: {label_col!r}")

    if time_col:
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        valid = parsed.notna()
        result.append(f"Timestamp parse rate: {valid.mean():.1%}")
        if valid.any():
            times = parsed[valid]
            result.append(f"Sample time range: {times.min()} -> {times.max()}")
            result.append(f"Unique minute bins in sample: {times.dt.floor('min').nunique():,}")
            result.append(f"Rows ordered by timestamp already: {bool(times.is_monotonic_increasing)}")
            # Label changes by consecutive 10-minute bins. This is a descriptive sample, not a split.
            if label_col:
                tmp = pd.DataFrame({"time": parsed, "label": df[label_col].astype("string").fillna("MISSING")})
                tmp = tmp[tmp["time"].notna()].sort_values("time")
                tmp["minute_bin"] = tmp["time"].dt.floor("10min")
                per_bin = tmp.groupby("minute_bin")["label"].agg(
                    lambda values: Counter(str(x).strip() for x in values).most_common(1)[0][0]
                )
                result.append(f"10-minute bins with labels: {len(per_bin):,}")
                result.append(f"Most-common label per 10-minute bin (top 20): {Counter(per_bin).most_common(20)}")
                transitions = int((per_bin != per_bin.shift()).sum() - (1 if len(per_bin) else 0))
                result.append(f"Changes in dominant 10-minute-bin label: {transitions}")
    else:
        result.append("WARNING: No timestamp column selected; inspect headers before temporal modeling.")

    if label_col:
        labels = df[label_col].astype("string").fillna("MISSING").str.strip()
        counts = labels.value_counts(dropna=False)
        result.append(f"Distinct labels in sample: {len(counts)}")
        result.append("Label counts in sample (top 30):")
        result.extend([f"  {str(label)!r}: {int(count)}" for label, count in counts.head(30).items()])
        if len(counts) == 1:
            result.append("WARNING: Only one label appears in this sampled portion.")
    else:
        result.append("WARNING: No attack label candidate selected.")

    # Candidate feature count is informational only; never automatically uses all columns.
    numeric_count = int(df.select_dtypes(include="number").shape[1])
    result.append(f"Columns inferred numeric by pandas: {numeric_count}")
    print(f"[OK] {dataset} / {path.name}: rows={len(df)}, time={time_col}, target={label_col}, labels={df[label_col].nunique() if label_col else 'unknown'}", flush=True)
    return result

def main():
    print("[START] Targeted temporal suitability check", flush=True)
    print(f"[INFO] Project root: {ROOT}", flush=True)
    print(f"[INFO] Maximum sample per file: {MAX_ROWS:,} rows", flush=True)
    report = [
        "TARGETED TEMPORAL SUITABILITY CHECK",
        f"Project root: {ROOT}",
        f"Sample cap: {MAX_ROWS:,} rows per file",
        "Source data is read-only. This is a suitability check, not a performance benchmark.",
        "Raw/processed duplicates are not treated as independent datasets.",
    ]
    for index, (dataset, path) in enumerate(FILES, 1):
        print(f"[FILE {index}/{len(FILES)}] {path.name}", flush=True)
        report.extend(inspect_file(dataset, path))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\n[DONE] Report saved to: {OUT_FILE}", flush=True)

if __name__ == "__main__":
    main()
