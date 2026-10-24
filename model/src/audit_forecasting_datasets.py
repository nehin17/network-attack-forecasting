"""Read-only audit of CICIDS, UNSW-NB15, and TON_IoT for temporal forecasting.
Run from project root:
  .\.venv\Scripts\python.exe -u model\src\audit_forecasting_datasets.py
Writes only model/results/dataset_audit/dataset_audit_report.txt; source data is untouched.
"""
from pathlib import Path
from collections import Counter
import csv, math, re
ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "model" / "results" / "dataset_audit"
REPORT_PATH = REPORT_DIR / "dataset_audit_report.txt"
CANDIDATES = {
    "CICIDS2017": [ROOT/"DATA"/"raw"/"CICID", ROOT/"DATA"/"processed"/"CICIDS"],
    "UNSW-NB15": [ROOT/"DATA"/"processed"/"UNSW", ROOT/"DATA"/"raw"/"UNSWNB", ROOT/"DATA"/"raw"/"UNSW"],
    "TON_IoT": [ROOT/"DATA"/"raw"/"TON_IoT"/"Network", ROOT/"DATA"/"processed"/"TON_IoT"/"Network", ROOT/"DATA"/"processed"/"TON_IoT"],
}
TIME_HINTS = ("timestamp","time","datetime","date_time","stime","ltime","starttime","start_time","event_time","ts")
LABEL_HINTS = ("label","attack","class","category","type","subcategory","attack_cat","attack_type","target")
ID_HINTS = {"flowid","srcip","dstip","sourceip","destinationip","id"}

def norm(s): return re.sub(r"[^a-z0-9]+","",str(s).strip().lower())
def detect(cols, hints):
    found=[c for c in cols if any(norm(h) in norm(c) for h in hints)]
    return sorted(found,key=lambda c:(0 if norm(c) in {"timestamp","label","attackcat","attacktype","stime","time","class"} else 1,len(c)))

def parse_time(v):
    v=str(v).strip()
    if not v: return None
    try:
        from datetime import datetime, timezone
        x=float(v)
        if 1_000_000_000 <= x <= 10_000_000_000:
            return datetime.fromtimestamp(x,tz=timezone.utc).replace(tzinfo=None)
    except Exception: pass
    try:
        import pandas as pd
        x=pd.to_datetime(v,errors="coerce",utc=True)
        if pd.isna(x): return None
        return x.to_pydatetime().replace(tzinfo=None)
    except Exception: return None

def inspect(path, dataset, max_rows=20000):
    r={"dataset":dataset,"path":str(path),"error":None,"rows_scanned":0,"columns":[],"time_cols":[],"time_stats":{},"label_cols":[],"label_stats":{},"numeric":[],"warnings":[]}
    try:
        with path.open("r",encoding="utf-8-sig",errors="replace",newline="") as f:
            reader=csv.reader(f)
            headers=next(reader,[])
            headers=[x.strip() for x in headers]
            r["columns"]=headers
            if not headers:
                r["error"]="Empty CSV"
                return r
            times=detect(headers,TIME_HINTS); labels=detect(headers,LABEL_HINTS)
            r["time_cols"]=times; r["label_cols"]=labels
            ix={c:i for i,c in enumerate(headers)}
            label_counts={c:Counter() for c in labels}; missing={c:0 for c in labels}
            time_vals={c:[] for c in times}; numeric_counts=Counter(); numeric_checked=Counter()
            for row in reader:
                if r["rows_scanned"] >= max_rows:
                    break
                if len(row)!=len(headers):
                    continue
                r["rows_scanned"]+=1
                for c in labels:
                    val=row[ix[c]].strip()
                    if not val or val.lower() in {"nan","none","null"}: missing[c]+=1
                    else: label_counts[c][val]+=1
                for c in times:
                    dt=parse_time(row[ix[c]])
                    if dt is not None: time_vals[c].append(dt)
                if r["rows_scanned"]<=5000:
                    excluded={norm(z) for z in times+labels}
                    for c in headers:
                        if norm(c) in excluded or norm(c) in ID_HINTS: continue
                        numeric_checked[c]+=1
                        try:
                            x=float(row[ix[c]])
                            if math.isfinite(x): numeric_counts[c]+=1
                        except Exception: pass
            r["sampled_rows"]=r["rows_scanned"]
            r["label_stats"]={c:{"counts":label_counts[c],"missing":missing[c]} for c in labels}
            r["numeric"]=[c for c in headers if numeric_checked[c] and numeric_counts[c] >= min(100,numeric_checked[c]//2)]
            for c,vals in time_vals.items():
                if vals:
                    r["time_stats"][c]={"parsed":len(vals),"sampled":r["rows_scanned"],"rate":len(vals)/max(1,r["rows_scanned"]),"min":min(vals).isoformat(sep=" "),"max":max(vals).isoformat(sep=" "),"unique_minutes":len(set(v.replace(second=0,microsecond=0) for v in vals)),"monotonic":all(vals[i]<=vals[i+1] for i in range(len(vals)-1))}
            if not times:r["warnings"].append("No obvious timestamp column detected.")
            elif not r["time_stats"]:r["warnings"].append("Timestamp-like columns found but sample values did not parse.")
            if not labels:r["warnings"].append("No obvious attack/label column detected.")
            r["warnings"].append(f"Sample-based profile: first {r['rows_scanned']} rows only; not a full-file class count.")
    except Exception as e:
        r["error"]=f"{type(e).__name__}: {e}"
    return r

def list_csvs(folders):
    found=[]; seen=set()
    for folder in folders:
           if folder.exists():
            for p in sorted(folder.rglob("*.csv")):
                k=str(p.resolve()).lower()
                if k not in seen: seen.add(k); found.append(p)
    return found

def main():
    print("[START] Read-only dataset audit",flush=True)
    print(f"[INFO] Project root: {ROOT}",flush=True)
    results=[]
    for dataset,folders in CANDIDATES.items():
        paths=list_csvs(folders)
        print(f"\n[DATASET] {dataset}: {len(paths)} CSV file(s)",flush=True)
        if not paths:
            results.append({"dataset":dataset,"path":"(no CSV found in candidate folders)","error":None,"rows":0,"columns":[],"time_cols":[],"time_stats":{},"label_cols":[],"label_stats":{},"numeric":[],"warnings":["Check configured folders."]})
        for p in paths:
            print(f"[SCAN] {p.relative_to(ROOT)}",flush=True)
            r=inspect(p,dataset, max_rows=20000); results.append(r)
            print(f"  [OK] sampled_rows={r.get('sampled_rows', 0)} cols={len(r['columns'])} numeric={len(r['numeric'])} timestamp={'yes' if r['time_stats'] else 'unresolved'} labels={'yes' if r['label_stats'] else 'unresolved'}",flush=True)
    lines=["READ-ONLY DATASET AUDIT: FUTURE ATTACK-TYPE FORECASTING",f"Project root: {ROOT}","Source datasets were not modified.","Detailed profiling uses at most 250,000 rows per CSV; total row counts are complete.",""]
    for r in results:
        lines += [f"DATASET: {r['dataset']}",f"FILE: {r['path']}",f"Rows sampled: {r.get('sampled_rows', 0)} (maximum 20,000)"]
        if r["error"]: lines += [f"ERROR: {r['error']}",""]; continue
        lines += [f"Columns ({len(r['columns'])}): {', '.join(r['columns'])}",f"Timestamp candidates: {r['time_cols'] or 'NONE'}"]
        for c,s in r["time_stats"].items():
            lines.append(f"  {c}: parse rate {s['rate']:.1%}; range {s['min']} -> {s['max']}; unique minutes in sample={s['unique_minutes']}; monotonic sampled={s['monotonic']}")
        lines += [f"Label candidates: {r['label_cols'] or 'NONE'}"]
        for c,obj in r["label_stats"].items():
            lines.append(f"  {c}: {len(obj['counts'])} distinct; missing={obj['missing']}")
            lines += [f"    {lab!r}: {n}" for lab,n in obj["counts"].most_common(20)]
        lines += [f"Numeric candidate features ({len(r['numeric'])}): {', '.join(r['numeric'][:100])}"]
        lines += [f"WARNING: {w}" for w in r["warnings"]]
if __name__ == "__main__":
    main()
    lines += [
        "",
        "PRELIMINARY GUIDANCE",
        "- Verify timestamps, labels, and attack coverage before choosing a dataset.",
        "- Preserve chronology; never shuffle overlapping windows across train/test.",
        "- Avoid target labels and direct identifiers as predictive features.",
        "- This is a preliminary audit, not a model-performance comparison.",
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[DONE] Report saved: {REPORT_PATH}", flush=True)

