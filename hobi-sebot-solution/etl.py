from scipy.spatial import cKDTree
import pickle
import pandas as pd
from constants import tree_path, data_path

df = pd.read_csv("data.csv")
tree = cKDTree(df.loc[:, df.columns.str.endswith("position")].values)

tree_path.write_bytes(pickle.dumps(tree))
data_path.write_bytes(pickle.dumps(df))