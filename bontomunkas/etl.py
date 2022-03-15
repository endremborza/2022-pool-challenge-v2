from scipy.spatial import cKDTree
import pickle
from constants import tree_path, data_path
import pandas as pd
from pathlib import Path
import numpy as np

pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]

df = (
    pd.read_csv(Path("data.csv"))
    .drop_duplicates(pos_cols)
    .dropna()
    .loc[:, lambda _df: _df.nunique() != 1]
)

for subject in df["subject"].unique():
    df_filtered = df[lambda _df: df["subject"] == subject].sort_values("msec").reset_index()

    position_data = df_filtered.loc[:, [*pos_cols]].values
    tree = cKDTree(position_data)

    (tree_path / subject).with_suffix(".pickle").write_bytes(pickle.dumps(tree))
    (data_path / subject).with_suffix(".pickle").write_bytes(pickle.dumps(df_filtered))