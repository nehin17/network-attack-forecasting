import os
import numpy as np
import pandas as pd

# ==============================================================================
# DIRECTORY CONFIGURATION
# ==============================================================================
BASE_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned"
CIC_DIR = os.path.join(BASE_DIR, "cic_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Final standardized 12-column schema
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

def find_csv_files(directory_path):
    """Recursively search for CSV files in directory and subdirectories."""
    csv_files = []
    for root, _, filenames in os.walk(directory_path):
        for f in filenames:
            if f.lower().endswith(".csv") and not f.startswith("."):
                csv_files.append(os.path.join(root, f))
    return csv_files

# ==============================================================================
# PROCESS & CLEAN CICIDS2017
# ==============================================================================
def process_cicids(data_dir):
    print("=" * 60)
    print("STEP 1: Locating CICIDS2017 CSV files...")
    print("=" * 60)
    
    csv_files = find_csv_files(data_dir)
    
    if not csv_files:
        raise FileNotFoundError(
            f"❌ No CSV files found inside: {data_dir}\n"
            f"Please verify that your raw files are in 'cic_data' and unzipped."
        )

    print(f"Found {len(csv_files)} CSV file(s):")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    print("\nSTEP 2: Reading and concatenating files...")
    df_list = []
    for file in csv_files:
        print(f"  Reading: {os.path.basename(file)}...")
        temp_df = pd.read_csv(file, low_memory=False)
        # Clean trailing and leading whitespace in header names
        temp_df.columns = temp_df.columns.str.strip()
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    initial_rows = len(df)
    print(f"\nTotal raw rows loaded: {initial_rows:,}")

    print("\nSTEP 3: Aligning column schema...")
    cic_mapping = {
        "Timestamp": "Timestamp",
        "Source Port": "Source_Port",
        "Destination Port": "Destination_Port",
        "Flow Duration": "Flow_Duration",
        "Total Fwd Packets": "Total_Fwd_Packets",
        "Total Backward Packets": "Total_Bwd_Packets",
        "Total Length of Fwd Packets": "Total_Len_Fwd_Packets",
        "Total Length of Bwd Packets": "Total_Len_Bwd_Packets",
        "Fwd IAT Mean": "Fwd_IAT_Mean",
        "Bwd IAT Mean": "Bwd_IAT_Mean",
        "Protocol": "Protocol",
        "Label": "Label"
    }

    df = df.rename(columns=cic_mapping)

    # Fill missing expected columns with 0 if any raw file missed them
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Retain strictly the target schema
    df = df[FINAL_COLUMNS].copy()

    print("\nSTEP 4: Cleaning missing values, infinities, and duplicates...")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    print("\nSTEP 5: Parsing timestamps and sorting chronologically...")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)
    df.sort_values(by="Timestamp", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("\nSTEP 6: Encoding Target Label to Binary (0 = BENIGN, 1 = Attack)...")
    df["Label"] = df["Label"].astype(str).str.strip().apply(
        lambda x: 0 if x.upper() == "BENIGN" else 1
    )

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
        df_cic = process_cicids(CIC_DIR)

        output_file = os.path.join(OUTPUT_DIR, "train_cicids2017_cleaned.csv")
        print(f"\nSTEP 7: Exporting to output folder...")
        df_cic.to_csv(output_file, index=False)

        print("\n" + "=" * 60)
        print("🎉 SUCCESS! CICIDS2017 processed and saved.")
        print(f"Location: {output_file}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error encountered: {e}")