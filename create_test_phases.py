import os
import pandas as pd
import numpy as np

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
TRAIN_FILE = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output/train_1min_scaled.csv"
OUTPUT_DIR = "/Users/pratibhachaudhary/Desktop/dataset cleaned/output"

PHASE1_OUTPUT = os.path.join(OUTPUT_DIR, "phase1_indistribution_test.csv")
PHASE2_OUTPUT = os.path.join(OUTPUT_DIR, "phase2_perturbed_test.csv")

def generate_test_phases(train_path):
    print(f"Loading scaled training dataset from: {train_path}...")
    df_train = pd.read_csv(train_path)
    
    feature_cols = [
        'flow_count', 'total_flow_duration', 'mean_flow_duration',
        'total_fwd_packets', 'total_bwd_packets', 'total_fwd_bytes',
        'total_bwd_bytes', 'mean_fwd_iat', 'mean_bwd_iat',
        'unique_src_ips', 'unique_dst_ports'
    ]

    # ==========================================
    # PHASE 1: IN-DISTRIBUTION TEST SET (20% Slice)
    # ==========================================
    print("\n--- Generating Phase 1: In-Distribution Test Set ---")
    # Take a 20% stratified/sample slice from the training set
    phase1_df = df_train.sample(frac=0.20, random_state=42).copy()
    phase1_df.sort_index(inplace=True)
    
    print(f"Phase 1 Created: {len(phase1_df):,} samples")
    print(f"  Normal (0): {(phase1_df['Label'] == 0).sum():,}")
    print(f"  Attack (1): {(phase1_df['Label'] == 1).sum():,}")
    
    phase1_df.to_csv(PHASE1_OUTPUT, index=False)
    print(f"Saved to: {PHASE1_OUTPUT}")

    # ==========================================
    # PHASE 2: PERTURBED STRESS-TEST SET (Noise Injection)
    # ==========================================
    print("\n--- Generating Phase 2: Perturbed Stress-Test Set ---")
    phase2_df = df_train.copy()
    
    # Apply Gaussian Noise Injection to numeric features
    noise_level = 0.15 # 15% noise variance
    np.random.seed(42)
    
    for col in feature_cols:
        std_dev = phase2_df[col].std()
        if std_dev == 0 or np.isnan(std_dev):
            std_dev = 1.0
            
        noise = np.random.normal(0, noise_level * std_dev, size=len(phase2_df))
        phase2_df[col] = phase2_df[col] + noise
        
    print(f"Phase 2 Created (Perturbed): {len(phase2_df):,} samples")
    print(f"  Applied Gaussian Noise (\u03c3 = {noise_level}) across {len(feature_cols)} features.")
    print(f"  Normal (0): {(phase2_df['Label'] == 0).sum():,}")
    print(f"  Attack (1): {(phase2_df['Label'] == 1).sum():,}")

    phase2_df.to_csv(PHASE2_OUTPUT, index=False)
    print(f"Saved to: {PHASE2_OUTPUT}")

    print("\nAll 3 Testing Phase Datasets are Ready!")

if __name__ == "__main__":
    generate_test_phases(TRAIN_FILE)