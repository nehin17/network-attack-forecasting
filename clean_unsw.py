import os
import glob
import pandas as pd
import numpy as np

INPUT_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned/unsw_data"
OUTPUT_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "unsw_nb15_cleaned_chronological.csv")

def find_raw_unsw_files(root_dir):
    raw_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and not file.startswith('.'):
                filename_lower = file.lower()
                if any(k in filename_lower for k in ['training', 'testing', 'list_events', 'features']):
                    continue
                raw_files.append(os.path.join(root, file))
    return raw_files

def clean_and_process_unsw_data(input_folder, output_path):
    all_files = find_raw_unsw_files(input_folder)
    print(f"Processing {len(all_files)} raw UNSW-NB15 flow files...")
    
    # Official 49 raw column names defined by ADFA UNSW-NB15 lab
    unsw_raw_headers = [
        'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
        'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
        'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
        'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
    ]

    dataframes = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"Reading: {filename}...")
        
        # Read raw files strictly using official 49 column positions
        df = pd.read_csv(file_path, names=unsw_raw_headers, header=None, low_memory=False, encoding_errors='replace')
        
        # If row 0 contains header string 'srcip', drop row 0
        if str(df.iloc[0, 0]).lower() == 'srcip':
            df = df.iloc[1:].copy()
            
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract required features directly from official column indices
    processed_df = pd.DataFrame()
    processed_df['Timestamp'] = pd.to_numeric(combined_df['Stime'], errors='coerce')
    processed_df['Source_IP'] = combined_df['srcip']
    processed_df['Destination_IP'] = combined_df['dstip']
    processed_df['Source_Port'] = pd.to_numeric(combined_df['sport'], errors='coerce')
    processed_df['Destination_Port'] = pd.to_numeric(combined_df['dsport'], errors='coerce')
    processed_df['Flow_Duration'] = pd.to_numeric(combined_df['dur'], errors='coerce') * 1e6 # Sec to Microsec
    processed_df['Total_Fwd_Packets'] = pd.to_numeric(combined_df['Spkts'], errors='coerce')
    processed_df['Total_Bwd_Packets'] = pd.to_numeric(combined_df['Dpkts'], errors='coerce')
    processed_df['Total_Len_Fwd_Packets'] = pd.to_numeric(combined_df['sbytes'], errors='coerce')
    processed_df['Total_Len_Bwd_Packets'] = pd.to_numeric(combined_df['dbytes'], errors='coerce')
    processed_df['Fwd_IAT_Mean'] = pd.to_numeric(combined_df['Sintpkt'], errors='coerce') * 1000.0 # msec to microsec
    processed_df['Bwd_IAT_Mean'] = pd.to_numeric(combined_df['Dintpkt'], errors='coerce') * 1000.0

    # Protocol mapping
    proto_map = {'tcp': 6, 'udp': 17, 'icmp': 1, 'hopopt': 0}
    def map_proto(val):
        v = str(val).strip().lower()
        if v in proto_map:
            return proto_map[v]
        try:
            return int(v)
        except ValueError:
            return 0
    processed_df['Protocol'] = combined_df['proto'].apply(map_proto)

    # DIRECT GROUND-TRUTH LABEL (Column Index 48 from UNSW Lab)
    processed_df['Label'] = pd.to_numeric(combined_df['Label'], errors='coerce').fillna(0).astype(int)

    # Convert Epoch Timestamp to Datetime and Sort
    processed_df['Timestamp'] = pd.to_datetime(processed_df['Timestamp'], unit='s', errors='coerce')
    processed_df.dropna(subset=['Timestamp'], inplace=True)
    processed_df.sort_values(by='Timestamp', ascending=True, inplace=True)
    processed_df.reset_index(drop=True, inplace=True)

    # Clean NaN/Inf
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = ['Source_Port', 'Destination_Port', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Bwd_Packets']
    processed_df.dropna(subset=num_cols, inplace=True)

    print(f"\nUNSW-NB15 Ground-Truth Distribution:")
    print(f"  Total Flows: {len(processed_df):,}")
    print(f"  Normal (0):  {(processed_df['Label'] == 0).sum():,}")
    print(f"  Attack (1):  {(processed_df['Label'] == 1).sum():,}")

    processed_df.to_csv(output_path, index=False)
    print(f"Successfully exported to: {output_path}")

if __name__ == "__main__":
    clean_and_process_unsw_data(INPUT_DIR, OUTPUT_FILE)