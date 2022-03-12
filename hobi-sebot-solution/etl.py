from scipy.spatial import cKDTree
import pickle
import pandas as pd
from constants import tree_path, data_path


pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]

df = (
    pd.read_csv("data.csv")
    .drop_duplicates(pos_cols)
    .dropna()
    .loc[:, lambda _df: _df.nunique() != 1]
)
tree = cKDTree(df.loc[:, df.columns.str.endswith("position")].values)

tree_path.write_bytes(pickle.dumps(tree))
data_path.write_bytes(pickle.dumps(df))
