import os
import glob
import pandas as pd
import numpy as np

INPUT_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned/TrafficLabelling"
OUTPUT_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "cicids2017_cleaned_chronological.csv")

def clean_col(col):
    return str(col).replace('\xa0', ' ').strip().replace('"', '').replace("'", "")

def identify_and_map_columns(df):
    cleaned_cols = {col: clean_col(col) for col in df.columns}
    df.rename(columns=cleaned_cols, inplace=True)
    
    mapping = {}
    for col in df.columns:
        cl = col.lower().replace('_', ' ').replace('-', ' ').strip()
        
        if 'timestamp' in cl or 'time stamp' in cl:
            mapping[col] = 'Timestamp'
        elif 'source ip' in cl or 'src ip' in cl:
            mapping[col] = 'Source_IP'
        elif 'destination ip' in cl or 'dst ip' in cl or 'dest ip' in cl:
            mapping[col] = 'Destination_IP'
        elif 'source port' in cl or 'src port' in cl:
            mapping[col] = 'Source_Port'
        elif 'destination port' in cl or 'dst port' in cl or 'dest port' in cl:
            mapping[col] = 'Destination_Port'
        elif 'flow duration' in cl:
            mapping[col] = 'Flow_Duration'
        elif 'total fwd packets' in cl or 'total forward packets' in cl:
            mapping[col] = 'Total_Fwd_Packets'
        elif 'total backward packets' in cl or 'total bwd packets' in cl:
            mapping[col] = 'Total_Bwd_Packets'
        elif 'total length of fwd packets' in cl or 'fwd header length' in cl:
            mapping[col] = 'Total_Len_Fwd_Packets'
        elif 'total length of bwd packets' in cl:
            mapping[col] = 'Total_Len_Bwd_Packets'
        elif 'fwd iat mean' in cl:
            mapping[col] = 'Fwd_IAT_Mean'
        elif 'bwd iat mean' in cl:
            mapping[col] = 'Bwd_IAT_Mean'
        elif cl == 'protocol':
            mapping[col] = 'Protocol'
        elif cl == 'label':
            mapping[col] = 'Label'

    df.rename(columns=mapping, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    
    target_cols = [
        'Timestamp', 'Source_IP', 'Destination_IP', 'Source_Port', 
        'Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets', 
        'Total_Bwd_Packets', 'Total_Len_Fwd_Packets', 'Total_Len_Bwd_Packets', 
        'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Protocol', 'Label'
    ]
    
    available_targets = [c for c in target_cols if c in df.columns]
    return df[available_targets].copy()

def find_all_csv_files(root_dir):
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and not file.startswith('.'):
                csv_files.append(os.path.join(root, file))
    return csv_files

def clean_and_process_cic_data(input_folder, output_path):
    all_files = find_all_csv_files(input_folder)
    print(f"Processing {len(all_files)} raw CICIDS2017 files...")
    dataframes = []

    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"Reading: {filename}...")
        try:
            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8', encoding_errors='replace')
        except Exception:
            df = pd.read_csv(file_path, low_memory=False, encoding='cp1252', encoding_errors='replace')
            
        df_mapped = identify_and_map_columns(df)
        dataframes.append(df_mapped)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()].copy()

    # Parse Timestamps & Sort
    combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'], format='mixed', errors='coerce')
    combined_df.dropna(subset=['Timestamp'], inplace=True)
    combined_df.sort_values(by='Timestamp', ascending=True, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    # Clean Numerics
    numeric_cols = [
        'Source_Port', 'Destination_Port', 'Flow_Duration',
        'Total_Fwd_Packets', 'Total_Bwd_Packets',
        'Total_Len_Fwd_Packets', 'Total_Len_Bwd_Packets',
        'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Protocol'
    ]
    for col in numeric_cols:
        if col in combined_df.columns:
            series = combined_df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            combined_df[col] = pd.to_numeric(series, errors='coerce')

    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined_df.dropna(subset=[c for c in numeric_cols if c in combined_df.columns], inplace=True)

    if 'Protocol' in combined_df.columns:
        combined_df['Protocol'] = combined_df['Protocol'].astype(int)

    # ORIGINAL GROUND-TRUTH ENCODING
    # Benign = 0, Any explicit attack string (DDoS, PortScan, Web Attack, etc.) = 1
    raw_labels = combined_df['Label'].astype(str).str.strip().str.upper()
    combined_df['Label'] = np.where(raw_labels.str.contains('BENIGN', na=False), 0, 1)

    master_cols = [
        'Timestamp', 'Source_IP', 'Destination_IP',
        'Source_Port', 'Destination_Port', 'Flow_Duration',
        'Total_Fwd_Packets', 'Total_Bwd_Packets',
        'Total_Len_Fwd_Packets', 'Total_Len_Bwd_Packets',
        'Fwd_IAT_Mean', 'Bwd_IAT_Mean', 'Protocol', 'Label'
    ]
    
    final_cols = [c for c in master_cols if c in combined_df.columns]
    combined_df = combined_df[final_cols]

    print(f"\nCICIDS2017 Ground-Truth Distribution:")
    print(f"  Total Flows: {len(combined_df):,}")
    print(f"  Normal (0):  {(combined_df['Label'] == 0).sum():,}")
    print(f"  Attack (1):  {(combined_df['Label'] == 1).sum():,}")

    combined_df.to_csv(output_path, index=False)
    print(f"Successfully exported to: {output_path}")

if __name__ == "__main__":
    clean_and_process_cic_data(INPUT_DIR, OUTPUT_FILE)