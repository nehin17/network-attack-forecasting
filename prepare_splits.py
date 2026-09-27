import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
INPUT_FILE = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output/cicids2017_1min_aggregated.csv"
OUTPUT_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_chronological_splits(input_path, output_dir):
    print(f"Loading 1-minute aggregated dataset: {input_path}...")
    df = pd.read_csv(input_path)
    
    # Feature columns for modeling (excluding Timestamp and Target Labels)
    feature_cols = [
        'flow_count', 'total_flow_duration', 'mean_flow_duration',
        'total_fwd_packets', 'total_bwd_packets', 'total_fwd_bytes',
        'total_bwd_bytes', 'mean_fwd_iat', 'mean_bwd_iat',
        'unique_src_ips', 'unique_dst_ports'
    ]
    
    total_steps = len(df)
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)

    print(f"\nTotal 1-minute time steps: {total_steps:,}")
    print(f"  Train Set (70%):  0 to {train_end:,} steps")
    print(f"  Val Set   (15%):  {train_end:,} to {val_end:,} steps")
    print(f"  Test Set  (15%):  {val_end:,} to {total_steps:,} steps")

    # Slice data chronologically
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # Fit StandardScaler ONLY on the training features
    print("\nFitting StandardScaler on TRAIN set only (preventing future leakage)...")
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    # Save fitted scaler for future inference
    scaler_path = os.path.join(output_dir, "scaler_train_only.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted scaler to: {scaler_path}")

    # Transform features across splits
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    # Export split CSVs
    train_path = os.path.join(output_dir, "train_1min_scaled.csv")
    val_path = os.path.join(output_dir, "val_1min_scaled.csv")
    test_path = os.path.join(output_dir, "test_1min_scaled.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSplits successfully created and exported!")
    print(f"  Train: {train_path} (Attack steps: {train_df['Label'].sum()})")
    print(f"  Val:   {val_path} (Attack steps: {val_df['Label'].sum()})")
    print(f"  Test:  {test_path} (Attack steps: {test_df['Label'].sum()})")

if __name__ == "__main__":
    create_chronological_splits(INPUT_FILE, OUTPUT_DIR)