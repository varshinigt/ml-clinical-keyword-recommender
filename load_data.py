# load_data.py
import pandas as pd, glob

def load_trials():
    files = sorted(glob.glob("data_parts/trials_part*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

if __name__ == "__main__":
    df = load_trials()
    print(df.shape)   # should print (50000, 7)