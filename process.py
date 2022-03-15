import pandas as pd
import numpy as np
import json
from pathlib import Path
import scipy.spatial.distance
from sklearn.neighbors import KDTree
import pickle

data = pd.read_pickle("data.pkl")
df = data.loc[:,["msec","x_position","y_position","z_position","subject","trial"]]
input_locations = json.loads(Path("input.json").read_text())
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
results_index = []
for query in input_locations:
    filt = (
        (df["subject"] == query["subject"])
        & (df["msec"] >= query["min_msec"])
        & (df["msec"] <= query["max_msec"])
    )
    dataset = df.loc[filt, pos_cols].reset_index()
    tree = KDTree(df.loc[filt, pos_cols], leaf_size=14)
    pos_arr = np.array([query[c] for c in pos_cols], ndmin=2)
    dist, ind = tree.query(pos_arr, k = 1)
    index = dataset.loc[ind[0][0],'index']
    results_index.append(index)
results = df.iloc[results_index, :][["msec", "subject", "trial"]].to_dict("records")
Path("output.json").write_text(json.dumps(results))