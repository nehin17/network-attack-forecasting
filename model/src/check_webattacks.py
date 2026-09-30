import pandas as pd

file = r"C:\Users\Neha\Desktop\semester 07\minor project (4)\network-attack-forecasting\DATA\raw\CICID\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"

df = pd.read_csv(file, encoding="latin1")

print("SHAPE:", df.shape)

print("\nDUPLICATES:", df.duplicated().sum())

print("\nLABELS:")
print(df[" Label"].value_counts(dropna=False))

print("\nTIMESTAMP:")
print(df[" Timestamp"].head())
print(df[" Timestamp"].tail())

print("\nMISSING VALUES — TOP 15:")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\nFIRST 5 ROWS:")
print(df.head())