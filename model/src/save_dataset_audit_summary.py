"""Generate a short audit report without rescanning any dataset CSVs.
Run from project root:
  .\.venv\Scripts\python.exe model\src\save_dataset_audit_summary.py
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "model" / "results" / "dataset_audit"
REPORT = OUT / "dataset_audit_report.txt"

content = """DATASET AUDIT SUMMARY — RECOVERED FROM THE COMPLETED TERMINAL SCAN

Important limitation:
The detailed audit script crashed after scanning the files, before writing its report.
This summary records only the facts visible in the terminal output supplied by the user.
It does NOT claim full-file class distributions or prove which dataset forecasts best.

CICIDS2017
- 16 CSV paths were scanned: the same eight named capture CSVs appeared under DATA/raw/CICID and DATA/processed/CICIDS.
- Each sampled CSV reported 85 columns, 63 candidate numeric columns, and timestamp and label candidates.
- Do not count the raw and processed copies as 16 independent captures; they appear to be duplicated capture names across two locations.
- Prior project audit showed the existing processed window set has 15 classes, but its original chronological test split contains only BENIGN and Bot.

UNSW-NB15
- 10 CSV paths were scanned across processed and raw folders.
- DATA/processed/UNSW/UNSW_NB15_testing-set.csv: 45 columns, 29 numeric candidates, timestamp and label candidates detected in the first 20,000 rows.
- DATA/processed/UNSW/UNSW_NB15_training-set.csv: 45 columns, 29 numeric candidates, timestamp and label candidates detected in the first 20,000 rows.
- Raw UNSW_NB15_1.csv through UNSW_NB15_4.csv: 49 columns; timestamp and label candidates were not resolved by the heuristic audit. This does not prove those files have no usable time or labels; inspect their headers and sample values directly.
- The raw and processed train/test CSVs should not be counted as independent datasets.

TON_IoT
- DATA/raw/TON_IoT/Network/train_test_network.csv: first 20,000 rows sampled; 44 columns, 12 numeric candidates, timestamp and label candidates detected.

DECISION STATUS
- All three remain candidates; the audit output alone does not identify a winner.
- Next, inspect the exact timestamp and target columns, label distribution over time, chronological order, and whether the standard train/test files are time-separated.
- Build windows within each source capture/file only, using past observations to predict a later target. Do not shuffle overlapping windows across train/test.
- Compare a majority baseline and temporal-feature XGBoost first; only add GRU/LSTM if the data supports meaningful sequential forecasting.
- Do not overwrite current-attack classification artifacts or existing datasets.
"""

OUT.mkdir(parents=True, exist_ok=True)
REPORT.write_text(content, encoding="utf-8")
print(f"[DONE] Saved recovered summary: {REPORT}")
print("No dataset CSVs were rescanned or modified.")
