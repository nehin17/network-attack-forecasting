import os
import pandas as pd
import numpy as np

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_FILE = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output/cicids2017_cleaned_chronological.csv"
OUTPUT_FILE = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output/cicids2017_1min_aggregated.csv"

def aggregate_temporal_traffic(input_path, output_path):
    print(f"Loading master clean CSV from: {input_path}...")
    df = pd.read_csv(input_path)
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    print(f"Loaded {len(df):,} total flows. Aggregating into 1-minute time windows...")

    aggregated_df = df.groupby(pd.Grouper(freq='1min')).agg(
        # Traffic volume metrics
        flow_count=('Flow_Duration', 'count'),
        total_flow_duration=('Flow_Duration', 'sum'),
        mean_flow_duration=('Flow_Duration', 'mean'),
        
        # Packet & byte volume
        total_fwd_packets=('Total_Fwd_Packets', 'sum'),
        total_bwd_packets=('Total_Bwd_Packets', 'sum'),
        total_fwd_bytes=('Total_Len_Fwd_Packets', 'sum'),
        total_bwd_bytes=('Total_Len_Bwd_Packets', 'sum'),
        
        # Delay / IAT averages
        mean_fwd_iat=('Fwd_IAT_Mean', 'mean'),
        mean_bwd_iat=('Bwd_IAT_Mean', 'mean'),
        
        # Endpoint activity
        unique_src_ips=('Source_IP', 'nunique'),
        unique_dst_ports=('Destination_Port', 'nunique'),
        
        # Target label: 1 if ANY attack flow occurred in this minute, else 0
        Label=('Label', lambda x: 1 if (x == 1).any() else 0),
        
        # Count of attack flows in this minute
        attack_flow_count=('Label', lambda x: (x == 1).sum())
    )

    # Fill unobserved time intervals with 0
    numeric_cols = aggregated_df.columns.drop(['Label'])
    aggregated_df[numeric_cols] = aggregated_df[numeric_cols].fillna(0)
    aggregated_df['Label'] = aggregated_df['Label'].fillna(0).astype(int)

    aggregated_df.reset_index(inplace=True)

    print("\nAggregation Complete!")
    print(f"Total 1-minute time steps: {len(aggregated_df):,}")
    print(f"Normal Minutes (0): {(aggregated_df['Label'] == 0).sum():,}")
    print(f"Attack Minutes (1): {(aggregated_df['Label'] == 1).sum():,}")

    print(f"\nSaving aggregated dataset to: {output_path}...")
    aggregated_df.to_csv(output_path, index=False)
    print("Successfully exported 1-minute aggregated file!")

if __name__ == "__main__":
    aggregate_temporal_traffic(INPUT_FILE, OUTPUT_FILE)