from pathlib import Path
from collections import Counter
import csv
import json
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "DATA" / "raw" / "CICID"
OUT_DIR = ROOT / "DATA" / "processed" / "future_attack_forecasting_cicids_v2"
REPORT_DIR = ROOT / "model" / "results" / "future_attack_forecasting_cicids_v2"

WINDOW_MINUTES = 10
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

CAPTURES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]


def parse_timestamps(series):
    """Parse timestamp formats while retaining a count of failures."""
    raw = series.astype("string").str.strip()

    try:
        parsed = pd.to_datetime(raw, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(raw, errors="coerce")

    missing = parsed.isna() & raw.notna() & raw.ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            raw.loc[missing],
            errors="coerce",
            format="%m/%d/%Y %H:%M:%S",
        )

    missing = parsed.isna() & raw.notna() & raw.ne("")
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            raw.loc[missing],
            errors="coerce",
        )

    return parsed, int(parsed.isna().sum())


def clean_label(value):
    value = str(value).strip()
    if value.lower() in {"", "nan", "none", "<na>"}:
        return "MISSING_LABEL"
    return value


def aggregate_capture(path):
    print(f"[READ] {path.name}", flush=True)
    started = time.time()

    df = pd.read_csv(
        path,
        low_memory=False,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    df.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in df.columns
    ]

    time_col = next(
        (c for c in df.columns if c.strip().lower() == "timestamp"),
        None,
    )
    label_col = next(
        (c for c in df.columns if c.strip().lower() == "label"),
        None,
    )

    if not time_col or not label_col:
        raise ValueError(
            f"Required Timestamp/Label columns missing in {path.name}"
        )

    original_rows = len(df)

    # FIX: remove rows that are completely empty across every column.
    # They contain no timestamp, label, or usable flow features.
    empty_mask = df.isna().all(axis=1)
    empty_strings_mask = df.astype("string").apply(
        lambda col: col.str.strip().eq("").fillna(True)
    ).all(axis=1)

    empty_mask = empty_mask | empty_strings_mask
    empty_rows = int(empty_mask.sum())

    if empty_rows:
        df = df.loc[~empty_mask].copy()
        print(
            f"[CLEAN] {path.name}: skipped {empty_rows:,} "
            "fully empty CSV rows",
            flush=True,
        )

    timestamps, bad_count = parse_timestamps(df[time_col])
    valid = timestamps.notna()

    invalid_nonempty_rows = int((~valid).sum())

    # Do not silently use rows with invalid timestamps.
    df = df.loc[valid].copy()
    df["_timestamp"] = timestamps.loc[valid].values
    df["_label"] = df[label_col].map(clean_label)

    if df.empty:
        raise ValueError(f"No valid timestamp rows remain in {path.name}")

    excluded = {
        time_col,
        label_col,
        "Flow ID",
        "Source IP",
        "Destination IP",
    }

    feature_cols = []

    for column in df.columns:
        if column.startswith("_") or column in excluded:
            continue

        if column.strip().lower() in {
            "source port",
            "destination port",
        }:
            continue

        numeric = pd.to_numeric(df[column], errors="coerce")

        if numeric.notna().any():
            df[column] = numeric.replace(
                [np.inf, -np.inf],
                np.nan,
            )
            feature_cols.append(column)

    if not feature_cols:
        raise ValueError(f"No numeric features found in {path.name}")

    df["_minute"] = pd.to_datetime(df["_timestamp"]).dt.floor("min")

    # Numeric flow features aggregated by minute.
    grouped = df.groupby("_minute", sort=True)
    features = grouped[feature_cols].mean()
    features["flow_count"] = grouped.size().astype(float)

    # Label each minute:
    # if attacks occur, use the most frequent non-BENIGN attack;
    # otherwise use BENIGN.
    def minute_target(values):
        labels = [str(value).strip() for value in values]
        attacks = [
            label for label in labels
            if label.upper() != "BENIGN"
        ]

        selected = attacks if attacks else labels
        counts = Counter(selected)
        highest = max(counts.values())

        return sorted(
            label for label, count in counts.items()
            if count == highest
        )[0]

    targets = grouped["_label"].agg(minute_target)

    minute_df = features.join(targets.rename("target")).sort_index()

    feature_names = list(features.columns)
    minute_df[feature_names] = (
        minute_df[feature_names]
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
    )

    minute_df["target"] = minute_df["target"].astype(str)

    class_counts = Counter(minute_df["target"])

    info = {
        "capture": path.name,
        "source_rows": int(original_rows),
        "empty_rows_skipped": empty_rows,
        "invalid_timestamp_rows_after_cleanup": invalid_nonempty_rows,
        "valid_timestamps": int(valid.sum()),
        "minute_bins": len(minute_df),
        "classes": dict(sorted(class_counts.items())),
        "first_minute": str(minute_df.index.min()),
        "last_minute": str(minute_df.index.max()),
        "feature_count": len(feature_names),
        "elapsed_seconds": round(time.time() - started, 2),
    }

    print(
        f"[AGG] {path.name}: source_rows={original_rows:,}; "
        f"empty_rows_skipped={empty_rows:,}; "
        f"valid_timestamps={int(valid.sum()):,}; "
        f"invalid_nonempty_rows={invalid_nonempty_rows:,}; "
        f"minutes={len(minute_df):,}; "
        f"classes={len(class_counts)}; "
        f"elapsed={info['elapsed_seconds']}s",
        flush=True,
    )

    return minute_df, feature_names, info


def make_windows(minute_df, feature_names):
    """Use 10 consecutive history minutes to predict the next minute."""
    if len(minute_df) <= WINDOW_MINUTES:
        return None

    x_values = minute_df[feature_names].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    y_values = minute_df["target"].to_numpy(dtype=str)
    times = minute_df.index

    xs, ys, metadata = [], [], []

    for target_idx in range(WINDOW_MINUTES, len(minute_df)):
        start_idx = target_idx - WINDOW_MINUTES

        expected = pd.date_range(
            start=times[start_idx],
            periods=WINDOW_MINUTES + 1,
            freq="min",
        )

        actual = times[start_idx:target_idx + 1]

        # Do not bridge missing minutes.
        if not actual.equals(pd.DatetimeIndex(expected)):
            continue

        xs.append(x_values[start_idx:target_idx])
        ys.append(y_values[target_idx])
        metadata.append(str(times[target_idx]))

    if not xs:
        return None

    return np.stack(xs), np.asarray(ys, dtype=str), metadata


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("[START] Build CICIDS future forecasting dataset", flush=True)
    print(f"[OUTPUT] {OUT_DIR}", flush=True)

    split_chunks = {
        "train": [],
        "val": [],
        "test": [],
    }

    metadata_rows = []
    capture_reports = []
    global_class_counts = Counter()
    global_features = None

    for index, name in enumerate(CAPTURES, start=1):
        print(f"[CAPTURE {index}/{len(CAPTURES)}]", flush=True)

        path = SOURCE_DIR / name

        if not path.exists():
            print(f"[WARN] Missing capture: {path}", flush=True)
            capture_reports.append({
                "capture": name,
                "status": "MISSING",
            })
            continue

        minute_df, feature_names, info = aggregate_capture(path)
        info["status"] = "OK"
        capture_reports.append(info)

        # The CICIDS captures are expected to share a feature schema.
        if global_features is None:
            global_features = feature_names.copy()
        else:
            # Align to the existing common feature schema.
            for column in global_features:
                if column not in minute_df.columns:
                    minute_df[column] = 0.0

            new_features = [
                column for column in feature_names
                if column not in global_features
            ]

            for column in new_features:
                minute_df = minute_df.drop(columns=[column])

            minute_df = minute_df.reindex(
                columns=global_features + ["target"],
                fill_value=0.0,
            )

        n = len(minute_df)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

        partitions = {
            "train": minute_df.iloc[:train_end],
            "val": minute_df.iloc[train_end:val_end],
            "test": minute_df.iloc[val_end:],
        }

        for split_name, partition in partitions.items():
            made = make_windows(partition, global_features)

            if made is None:
                print(
                    f"[WARN] No windows for {name} / {split_name}",
                    flush=True,
                )
                continue

            X, y, target_times = made
            split_chunks[split_name].append((X, y))

            for target, timestamp in zip(y, target_times):
                metadata_rows.append({
                    "capture": name,
                    "target_timestamp": timestamp,
                    "split": split_name,
                    "target": str(target),
                })

            global_class_counts.update(y.tolist())

            print(
                f"[WINDOWS] {split_name}: {len(y):,}",
                flush=True,
            )

    if global_features is None:
        raise RuntimeError("No usable CICIDS captures found.")

    split_counts = {}

    for split_name, chunks in split_chunks.items():
        if chunks:
            X = np.concatenate([part[0] for part in chunks])
            y = np.concatenate([part[1] for part in chunks])
        else:
            X = np.empty(
                (0, WINDOW_MINUTES, len(global_features)),
                dtype=np.float32,
            )
            y = np.empty((0,), dtype=str)

        output_file = OUT_DIR / f"{split_name}_windows.npz"

        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            feature_names=np.asarray(global_features, dtype=str),
        )

        split_counts[split_name] = dict(sorted(Counter(y.tolist()).items()))

        print(
            f"[SAVE] {output_file.name}: X={X.shape}, y={y.shape}",
            flush=True,
        )

    metadata_file = OUT_DIR / "window_metadata.csv"

    with metadata_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "capture",
                "target_timestamp",
                "split",
                "target",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    config = {
        "task": (
            "Predict the target label in the next consecutive minute "
            "from the preceding 10 consecutive minute bins."
        ),
        "window_minutes": WINDOW_MINUTES,
        "train_fraction_per_capture": TRAIN_FRAC,
        "validation_fraction_per_capture": VAL_FRAC,
        "test_fraction_per_capture": 1.0 - TRAIN_FRAC - VAL_FRAC,
        "label_rule": (
            "Most frequent non-BENIGN label per minute if any attack "
            "flows occur; otherwise BENIGN."
        ),
        "features": global_features,
        "class_counts_all_splits": dict(sorted(global_class_counts.items())),
        "split_class_counts": split_counts,
        "timestamp_policy": (
            "Completely empty CSV records are excluded. Rows with invalid "
            "timestamps are reported and excluded from temporal aggregation."
        ),
    }

    (OUT_DIR / "dataset_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    (REPORT_DIR / "capture_summary.json").write_text(
        json.dumps(capture_reports, indent=2),
        encoding="utf-8",
    )

    lines = [
        "CICIDS FUTURE-ATTACK-TYPE DATASET SUMMARY",
        f"Dataset directory: {OUT_DIR}",
        f"History: {WINDOW_MINUTES} consecutive minutes",
        "Target: next consecutive minute",
        "Split: chronological within each capture, 70% / 15% / 15%",
        "",
        "SPLIT CLASS COUNTS:",
    ]

    for split_name, counts in split_counts.items():
        lines.append(f"{split_name.upper()}:")
        lines.extend(
            f"  {label}: {count:,}"
            for label, count in counts.items()
        )

    lines.extend(["", "CAPTURE TIMESTAMP SUMMARY:"])

    for info in capture_reports:
        lines.append(
            f"{info.get('capture')}: status={info.get('status')}; "
            f"empty_rows_skipped={info.get('empty_rows_skipped', 'n/a')}; "
            f"invalid_timestamp_rows={info.get('invalid_timestamp_rows_after_cleanup', 'n/a')}; "
            f"minutes={info.get('minute_bins', 'n/a')}"
        )

    summary_path = REPORT_DIR / "dataset_build_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[DONE] Summary: {summary_path}", flush=True)
    print(f"[DONE] Dataset: {OUT_DIR}", flush=True)
    print("[NEXT] Proceed to baseline and XGBoost training.", flush=True)


if __name__ == "__main__":
    main()