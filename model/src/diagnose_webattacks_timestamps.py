from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "DATA/raw/CICID/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

def main():
    print("[START] WebAttacks timestamp diagnosis", flush=True)
    df = pd.read_csv(path, low_memory=False, encoding="utf-8-sig", encoding_errors="replace", usecols=lambda c: c.strip().lower() == "timestamp")
    col = df.columns[0]
    s = df[col].astype("string").str.strip()
    print(f"[INFO] Column={col!r}, rows={len(s):,}", flush=True)
    for label, kwargs in [
        ("mixed", {"format": "mixed"}),
        ("mixed dayfirst", {"format": "mixed", "dayfirst": True}),
        ("month-first explicit", {"format": "%m/%d/%Y %H:%M:%S"}),
        ("day-first explicit", {"format": "%d/%m/%Y %H:%M:%S"}),
        ("month-first no seconds", {"format": "%m/%d/%Y %H:%M"}),
        ("day-first no seconds", {"format": "%d/%m/%Y %H:%M"}),
    ]:
        parsed = pd.to_datetime(s, errors="coerce", **kwargs)
        print(f"[TEST] {label}: parsed={parsed.notna().sum():,}; failed={parsed.isna().sum():,}", flush=True)
    parsed = pd.to_datetime(s, errors="coerce", format="mixed")
    failed = s[parsed.isna()]
    print("[FAILED EXAMPLES]")
    for v in failed.drop_duplicates().head(30).tolist():
        print(repr(str(v)))
    print("[COMMON FAILED VALUES]")
    print(failed.value_counts(dropna=False).head(20).to_string())
    print("[VALID EXAMPLES]")
    print(s[parsed.notna()].head(10).to_list())
    print("[DONE] Diagnosis only; no files modified.", flush=True)

if __name__ == "__main__":
    main()
