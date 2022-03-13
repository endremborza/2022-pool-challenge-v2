from scipy.spatial import cKDTree
import pickle
import pandas as pd
from constants import data_path
from pathlib import Path
import numpy as np


pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]

df = (
    pd.read_csv(Path("data.csv"))
    .drop_duplicates(pos_cols)
    .dropna()
    .loc[:, lambda _df: _df.nunique() != 1]
)

data_path.write_bytes(pickle.dumps(df))
