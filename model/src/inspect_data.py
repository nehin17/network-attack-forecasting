import pandas as pd
import glob
import os

folders = {
     "CICIDS2017": r"C:\Users\Neha\Desktop\semester 07\minor project (4)\network-attack-forecasting\DATA\raw\CICID",
     "UNSW-NB15": r"C:\Users\Neha\Desktop\semester 07\minor project (4)\network-attack-forecasting\DATA\raw\UNSWNB"
}

for dataset, folder in folders.items():
    print("\n" + "=" * 70)
    print(dataset)
    print("=" * 70)

    files = glob.glob(os.path.join(folder, "*.csv"))

    for file in files:
        print(f"\n--- {os.path.basename(file)} ---")

        try:
            # Read in chunks so huge files don't fill RAM
            chunks = []
            rows = 0
            duplicates = 0
            missing = None
            labels = {}

            for chunk in pd.read_csv(
                file,
                encoding="latin1",
                low_memory=False,
                chunksize=100000
            ):
                rows += len(chunk)

                if missing is None:
                    missing = chunk.isna().sum()

                else:
                    missing += chunk.isna().sum()

                if "Label" in chunk.columns:
                    counts = chunk["Label"].value_counts()
                    for label, count in counts.items():
                        labels[label] = labels.get(label, 0) + count

                if "attack_cat" in chunk.columns:
                    counts = chunk["attack_cat"].value_counts()
                    for label, count in counts.items():
                        labels[label] = labels.get(label, 0) + count

                duplicates += chunk.duplicated().sum()

            # Read only header
            header = pd.read_csv(
                file,
                encoding="latin1",
                nrows=0
            )

            print("Rows:", rows)
            print("Columns:", len(header.columns))
            print("Timestamp column:", 
                  [c for c in header.columns if "time" in c.lower()])
            print("Label columns:",
                  [c for c in header.columns
                   if "label" in c.lower() or "attack" in c.lower()])
            print("Duplicate rows detected in chunks:", duplicates)

            print("\nTop missing-value columns:")
            print(missing[missing > 0].sort_values(ascending=False).head(10))

            if labels:
                print("\nLabels:")
                for label, count in sorted(labels.items(), key=lambda x: -x[1]):
                    print(f"  {label}: {count}")

        except Exception as e:
            print("ERROR:", e)