import os
import numpy as np
import pandas as pd

# ==============================================================================
# DIRECTORY CONFIGURATION
# ==============================================================================
BASE_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned"
UNSW_DIR = os.path.join(BASE_DIR, "unsw_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

FINAL_COLUMNS = [
    "Timestamp",
    "Source_Port",
    "Destination_Port",
    "Flow_Duration",
    "Total_Fwd_Packets",
    "Total_Bwd_Packets",
    "Total_Len_Fwd_Packets",
    "Total_Len_Bwd_Packets",
    "Fwd_IAT_Mean",
    "Bwd_IAT_Mean",
    "Protocol",
    "Label"
]

def find_data_csv_files(directory_path):
    """Search for data CSV files and exclude feature documentation files."""
    csv_files = []
    for root, _, filenames in os.walk(directory_path):
        for f in filenames:
            # Exclude feature definition/metadata CSV files
            if f.lower().endswith(".csv") and not f.startswith(".") and "features" not in f.lower():
                csv_files.append(os.path.join(root, f))
    return csv_files

def read_csv_with_encoding_fallback(file_path):
    """Attempt reading CSV with UTF-8 first, falling back to Latin-1/CP1252."""
    for encoding in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
            return df
        except UnicodeDecodeError:
            continue
    # Fallback with error replacement
    return pd.read_csv(file_path, low_memory=False, encoding="utf-8", encoding_errors="replace")

# ==============================================================================
# PROCESS & CLEAN UNSW-NB15
# ==============================================================================
def process_unsw(data_dir):
    print("=" * 60)
    print("STEP 1: Locating UNSW-NB15 data CSV files...")
    print("=" * 60)
    
    csv_files = find_data_csv_files(data_dir)
    
    if not csv_files:
        raise FileNotFoundError(
            f"❌ No UNSW data CSV files found inside: {data_dir}"
        )

    print(f"Found {len(csv_files)} traffic data CSV file(s) (excluded metadata):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    print("\nSTEP 2: Reading and concatenating files...")
    df_list = []
    for file in csv_files:
        print(f"  Reading: {os.path.basename(file)}...")
        temp_df = read_csv_with_encoding_fallback(file)
        temp_df.columns = temp_df.columns.astype(str).str.strip()
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    initial_rows = len(df)
    print(f"\nTotal raw rows loaded: {initial_rows:,}")

    print("\nSTEP 3: Aligning column schema to match CICIDS2017...")
    unsw_mapping = {
        "Stime": "Timestamp",
        "sport": "Source_Port",
        "dsport": "Destination_Port",
        "dur": "Flow_Duration",
        "spkts": "Total_Fwd_Packets",
        "dpkts": "Total_Bwd_Packets",
        "sbytes": "Total_Len_Fwd_Packets",
        "dbytes": "Total_Len_Bwd_Packets",
        "sinpkt": "Fwd_IAT_Mean",
        "dinpkt": "Bwd_IAT_Mean",
        "proto": "Protocol",
        "label": "Label"
    }

    df = df.rename(columns=unsw_mapping)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
    else:
        df["Timestamp"] = pd.Timestamp.now()

    for port_col in ["Source_Port", "Destination_Port"]:
        if port_col in df.columns:
            df[port_col] = pd.to_numeric(df[port_col], errors="coerce").fillna(0).astype(int)
        else:
            df[port_col] = 0

    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[FINAL_COLUMNS].copy()

    print("\nSTEP 4: Cleaning missing values, infinities, and duplicates...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    print("\nSTEP 5: Sorting chronologically...")
    df.sort_values(by="Timestamp", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nSTEP 6: Standardizing Target Label (Binary 0 or 1)...")
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)

    clean_rows = len(df)
    print(f"\nCleaning Complete!")
    print(f"  - Initial Rows: {initial_rows:,}")
    print(f"  - Clean Rows  : {clean_rows:,}")
    print(f"  - Removed     : {initial_rows - clean_rows:,} invalid/duplicate rows")

    return df

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        df_unsw = process_unsw(UNSW_DIR)

        output_file = os.path.join(OUTPUT_DIR, "test_unswnb15_cleaned.csv")
        print(f"\nSTEP 7: Exporting to output folder...")
        df_unsw.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("🎉 SUCCESS! UNSW-NB15 processed and saved.")
        print(f"Location: {output_file}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error encountered: {e}")