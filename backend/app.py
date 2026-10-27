from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]

# Existing CICIDS2017 future-forecast model (unchanged).
FORECAST_MODEL_PATH = ROOT / "model/artifacts/cicids_binary_forecast/final_binary_forecaster.joblib"
CICIDS_RAW_CANDIDATES = [ROOT / "DATA/raw/CICID", ROOT / "DATA/raw/CICIDS"]
CICIDS_CAPTURES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
]

# Existing TON_IoT multiclass XGBoost model. Rebuild only the training-time
# preprocessing; the classifier itself is loaded and is NOT retrained.
TONIOT_DATA_PATH = ROOT / "DATA/raw/TON_IoT/Network/train_test_network.csv"
TONIOT_MODEL_PATH = ROOT / "model/results/ton_iot/toniot_xgboost_baseline.json"
TONIOT_DROP_COLUMNS = ["type", "label", "src_ip", "dst_ip"]

app = FastAPI(title="NetGuard Forecast API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_cicids_raw_dir():
    for path in CICIDS_RAW_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "CICIDS raw CSVs not found under DATA/raw/CICID or DATA/raw/CICIDS"
    )


@lru_cache(maxsize=1)
def get_forecast_bundle():
    if not FORECAST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved forecast model not found: {FORECAST_MODEL_PATH}")
    return joblib.load(FORECAST_MODEL_PATH)


def aggregate_cicids(path, feature_columns):
    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", encoding_errors="replace")
    except TypeError:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    time_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if time_col is None or label_col is None:
        raise ValueError(f"{path.name}: Timestamp/Label columns missing")
    raw_ts = df[time_col].astype("string").str.strip()
    try:
        ts = pd.to_datetime(raw_ts, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        ts = pd.to_datetime(raw_ts, errors="coerce")
    bad = ts.isna() & raw_ts.notna() & raw_ts.ne("")
    if bad.any():
        ts.loc[bad] = pd.to_datetime(raw_ts[bad], errors="coerce", dayfirst=False)
    bad = ts.isna() & raw_ts.notna() & raw_ts.ne("")
    if bad.any():
        ts.loc[bad] = pd.to_datetime(
            raw_ts[bad], errors="coerce", format="%m/%d/%Y %H:%M:%S"
        )
    labels = df[label_col].astype("string").str.strip()
    valid = ts.notna() & labels.notna() & labels.ne("")
    df = df.loc[valid].copy()
    df["_minute"] = ts.loc[valid].dt.floor("min").values
    df["_label"] = labels.loc[valid].astype(str).values
    excluded = {
        time_col, label_col, "Flow ID", "Source IP", "Destination IP",
        "Source_IP", "Destination_IP", "src_ip", "dst_ip",
    }
    numeric = []
    for col in df.columns:
        if col.startswith("_") or col in excluded:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            df[col] = vals.replace([np.inf, -np.inf], np.nan)
            numeric.append(col)
    grouped = df.groupby("_minute", sort=True)
    minute = grouped[numeric].mean()
    minute["flow_count"] = grouped.size().astype(float)

    def is_attack(value):
        return str(value).strip().upper() not in {
            "BENIGN", "NORMAL", "0", "NONE", "NAN", ""
        }

    target = grouped["_label"].apply(
        lambda values: int(any(is_attack(value) for value in values))
    ).astype(np.int8)
    minute = (
        minute.replace([np.inf, -np.inf], np.nan)
        .reindex(columns=feature_columns)
        .ffill()
        .fillna(0.0)
    )
    return minute.sort_index(), target.reindex(minute.index).fillna(0).astype(np.int8)


@lru_cache(maxsize=1)
def make_forecast_sample():
    bundle = get_forecast_bundle()
    columns = list(bundle["feature_columns"])
    folder = get_cicids_raw_dir()
    fallback = None
    for filename in CICIDS_CAPTURES:
        path = folder / filename
        if not path.exists():
            continue
        minute, target = aggregate_cicids(path, columns)
        n = len(minute)
        test_start = int(n * 0.85)
        values = minute.to_numpy(dtype=np.float32)
        labels, times = target.to_numpy(), minute.index
        for end in range(max(test_start + 10, 10), n):
            start = end - 10
            expected = pd.date_range(times[start], periods=11, freq="min")
            observed = times[start:end + 1]
            if len(observed) != 11 or not np.array_equal(observed.asi8, expected.asi8):
                continue
            sample = {
                "sequence": values[start:end].tolist(),
                "actual": "ATTACK" if int(labels[end]) else "BENIGN",
                "capture": filename,
                "history_start": times[start].isoformat(),
                "history_end": times[end - 1].isoformat(),
                "target_time": times[end].isoformat(),
            }
            if fallback is None:
                fallback = sample
            if int(labels[end]) == 1:
                return sample
    if fallback:
        return fallback
    raise RuntimeError("No consecutive 10-minute history + next-minute target found in test split")


@lru_cache(maxsize=1)
def get_toniot_bundle():
    if not TONIOT_DATA_PATH.exists():
        raise FileNotFoundError(f"TON_IoT dataset not found: {TONIOT_DATA_PATH}")
    if not TONIOT_MODEL_PATH.exists():
        raise FileNotFoundError(f"TON_IoT XGBoost model not found: {TONIOT_MODEL_PATH}")

    # Reproduce experiment_toniot_classifier.py's exact data preparation.
    df = pd.read_csv(TONIOT_DATA_PATH, low_memory=False)
    df = df.drop_duplicates().reset_index(drop=True)
    if "type" not in df.columns:
        raise ValueError("TON_IoT dataset has no 'type' target column")
    X = df.drop(columns=TONIOT_DROP_COLUMNS, errors="ignore").copy()
    y = df["type"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    cat_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    num_cols = [column for column in X.columns if column not in cat_cols]

    X_train = X_train.copy()
    X_test = X_test.copy()
    for column in cat_cols:
        X_train[column] = X_train[column].fillna("__MISSING__").astype(str)
        X_test[column] = X_test[column].fillna("__MISSING__").astype(str)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = encoder.transform(X_test[cat_cols])

    for column in num_cols:
        X_train[column] = pd.to_numeric(X_train[column], errors="coerce")
        X_test[column] = pd.to_numeric(X_test[column], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

    classes = sorted(y_train.unique())
    class_to_id = {name: index for index, name in enumerate(classes)}

    model = XGBClassifier()
    model.load_model(str(TONIOT_MODEL_PATH))

    # Test one model prediction against the model's expected class count.
    if len(classes) != int(model.n_classes_):
        raise ValueError(
            f"TON_IoT class mapping mismatch: dataset has {len(classes)} classes, "
            f"model has {model.n_classes_}"
        )

    return {
        "model": model,
        "encoder": encoder,
        "imputer": imputer,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "feature_columns": list(X.columns),
        "classes": classes,
        "X_test": X_test,
        "y_test": y_test.reset_index(drop=True),
        "y_test_ids": y_test.map(class_to_id).to_numpy(),
    }


@app.get("/")
def root():
    return {"service": "NetGuard Forecast API", "health": "/api/health", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "toniot_model_available": TONIOT_MODEL_PATH.exists(),
        "toniot_dataset_available": TONIOT_DATA_PATH.exists(),
        "forecast_model_available": FORECAST_MODEL_PATH.exists(),
        "cicids_dataset_available": any(path.exists() for path in CICIDS_RAW_CANDIDATES),
    }


@app.get("/api/detection/sample")
def detection_sample(index: int = 0):
    try:
        bundle = get_toniot_bundle()
        count = len(bundle["X_test"])
        if index < 0 or index >= count:
            raise HTTPException(status_code=400, detail=f"index must be between 0 and {count - 1}")

        row = bundle["X_test"].iloc[[index]]
        probabilities = bundle["model"].predict_proba(row)[0]
        predicted_id = int(np.argmax(probabilities))
        predicted_class = bundle["classes"][predicted_id]
        actual_class = str(bundle["y_test"].iloc[index])
        confidence = float(probabilities[predicted_id])

        # Explainable, rule-based severity from classifier confidence and predicted class.
        # This is a display heuristic, not a separately trained severity model.
        benign_names = {"normal", "benign", "none", "0"}
        is_attack = predicted_class.strip().lower() not in benign_names
        severity = (
            "HIGH" if is_attack and confidence >= 0.85
            else "MEDIUM" if is_attack and confidence >= 0.60
            else "LOW" if is_attack
            else "INFO"
        )
        risk_score = round(confidence * 100, 2) if is_attack else round((1 - confidence) * 100, 2)

        return {
            "prediction": predicted_class,
            "predicted_class": predicted_class,
            "actual_class": actual_class,
            "correct": predicted_class == actual_class,
            "confidence": round(confidence, 6),
            "class_probabilities": {
                bundle["classes"][i]: round(float(probabilities[i]), 6)
                for i in range(len(bundle["classes"]))
            },
            "is_attack": is_attack,
            "severity": severity,
            "risk_score": risk_score,
            "model": "TON_IoT XGBoost multiclass classifier",
            "sample_index": index,
            "feature_count": len(bundle["feature_columns"]),
            "source": "TON_IoT held-out test split",
            "note": "Offline dataset replay, not live traffic. Severity and risk are rule-based display scores.",
        }
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/forecast/sample")
def forecast_sample():
    try:
        bundle = get_forecast_bundle()
        sample = make_forecast_sample()
        seq = np.asarray(sample["sequence"], dtype=np.float32)
        expected = (int(bundle.get("history_minutes", 10)), len(bundle["feature_columns"]))
        if seq.shape != expected:
            raise ValueError(f"Expected sequence {expected}; got {seq.shape}")
        X = np.concatenate(
            [seq[-1], seq.mean(axis=0), seq.std(axis=0), seq[-1] - seq[0]]
        ).reshape(1, -1)
        probability = float(bundle["model"].predict_proba(X)[0, 1])
        threshold = float(bundle["threshold"])
        prediction = "ATTACK" if probability >= threshold else "BENIGN"
        return {
            "prediction": prediction,
            "attack_probability": round(probability, 6),
            "confidence": round(probability if prediction == "ATTACK" else 1 - probability, 6),
            "threshold": threshold,
            "model": bundle.get("selected_model", "xgboost"),
            "history_minutes": expected[0],
            "feature_count": expected[1],
            "horizon_minutes": 1,
            "actual_next_minute": sample["actual"],
            "capture": sample["capture"],
            "history_start": sample["history_start"],
            "history_end": sample["history_end"],
            "target_time": sample["target_time"],
            "source": "CICIDS2017 chronological test split",
            "note": "Offline test sample, not live traffic. The recall-oriented threshold can produce false positives.",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/traffic/sample")
def traffic_sample():
    """Return real, minute-binned flow statistics from the CICIDS2017 capture used by the forecast sample."""
    try:
        sample = make_forecast_sample()
        path = get_cicids_raw_dir() / sample["capture"]
        try:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", encoding_errors="replace")
        except TypeError:
            df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig")
        df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
        time_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if time_col is None or label_col is None:
            raise ValueError("CICIDS capture is missing Timestamp or Label columns")
        ts = pd.to_datetime(df[time_col].astype("string").str.strip(), errors="coerce")
        if ts.isna().any():
            ts = ts.fillna(pd.to_datetime(df[time_col].astype("string").str.strip(), errors="coerce", dayfirst=False))
        df = df.loc[ts.notna()].copy()
        df["_minute"] = ts.loc[df.index].dt.floor("min")
        labels = df[label_col].astype("string").str.strip().str.upper()
        df["_attack"] = (~labels.isin(["BENIGN", "NORMAL", "0", "NONE", ""])).astype(int)
        end = pd.Timestamp(sample["target_time"])
        start = end - pd.Timedelta(minutes=8)
        df = df[(df["_minute"] >= start) & (df["_minute"] <= end)]
        if df.empty:
            raise ValueError("No flow records found around the selected forecast window")
        grouped = df.groupby("_minute", sort=True)
        minutes = []
        for stamp, group in grouped:
            inbound_col = next((c for c in group.columns if c.lower() in {"total length of fwd packets", "totlen_fwd_pkts"}), None)
            outbound_col = next((c for c in group.columns if c.lower() in {"total length of bwd packets", "totlen_bwd_pkts"}), None)
            inbound = pd.to_numeric(group[inbound_col], errors="coerce").fillna(0).sum() if inbound_col else 0
            outbound = pd.to_numeric(group[outbound_col], errors="coerce").fillna(0).sum() if outbound_col else 0
            minutes.append({
                "time": stamp.strftime("%H:%M"),
                "inbound": round(float(inbound) / 1024, 2),
                "outbound": round(float(outbound) / 1024, 2),
                "flows": int(len(group)),
                "risk": round(float(group["_attack"].mean()) * 100, 1),
            })
        protocol_col = next((c for c in df.columns if c.lower() == "protocol"), None)
        protocol_names = {6: "TCP", 17: "UDP", 1: "ICMP", 58: "ICMPv6"}
        if protocol_col:
            protocol_values = pd.to_numeric(df[protocol_col], errors="coerce")
            counts = protocol_values.map(lambda x: protocol_names.get(int(x), f"Protocol {int(x)}") if pd.notna(x) else "Other").value_counts()
        else:
            counts = pd.Series({"Unknown": len(df)})
        total = int(counts.sum()) or 1
        protocols = [{"name": str(name), "value": round(float(count) * 100 / total, 1)} for name, count in counts.head(4).items()]
        return {
            "source": "CICIDS2017 offline capture replay",
            "capture": sample["capture"],
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "minutes": minutes,
            "protocols": protocols,
            "total_flows": int(len(df)),
            "attack_flows": int(df["_attack"].sum()),
            "risk_score": round(float(df["_attack"].mean()) * 100, 1),
            "note": "Dataset replay only; these are not live network measurements.",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
