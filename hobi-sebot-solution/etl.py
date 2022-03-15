from scipy.spatial import cKDTree
import pickle
import pandas as pd
from constants import tree_path, data_path
from pathlib import Path


pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]

df = (
    pd.read_csv(Path("data.csv"))
    .dropna()
    .loc[:, lambda _df: _df.nunique() != 1]
    .reset_index()
)

tree = cKDTree(df.loc[:, pos_cols].values)

tree_path.write_bytes(pickle.dumps(tree))

data_path.write_bytes(pickle.dumps(df))
