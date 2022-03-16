from scipy.spatial import cKDTree
import pickle
import pandas as pd
from constants import tree_path, data_path
from pathlib import Path

df = (
    pd.read_csv(Path("data.csv"))
    .dropna()
    .loc[:, lambda _df: _df.nunique() != 1]
).copy()


data_path.write_bytes(pickle.dumps(df))