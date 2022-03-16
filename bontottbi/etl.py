import pickle
from constants import data_path
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("data.csv")).dropna().loc[:, lambda _df: _df.nunique() != 1]

for subject in df["subject"].unique():
    df_filtered = (
        df[lambda _df: _df["subject"] == subject]
        .sort_values("msec")
        .reset_index(drop=True)
        .copy()
    )
    (data_path / subject).with_suffix(".pickle").write_bytes(pickle.dumps(df_filtered))
