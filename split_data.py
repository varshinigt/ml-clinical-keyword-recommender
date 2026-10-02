# split_data.py
import pandas as pd, numpy as np, os

df = pd.read_parquet("data/trials.parquet")
os.makedirs("data_parts", exist_ok=True)

for i, part in enumerate(np.array_split(df, 4), start=1):
    part.to_parquet(f"data_parts/trials_part{i}.parquet", compression="zstd")
    print(f"part {i}: {len(part)} rows")