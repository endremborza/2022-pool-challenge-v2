from constants import data_path, tree_path
import pickle
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

df = pickle.loads(data_path.read_bytes())
input_locations = json.loads(Path("input.json").read_text())
res_cols = ["msec", "subject", "trial"]
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]


def get_valid_indexes(query: dict):
    return (
        (df["subject"] == query["subject"])
        & (df["msec"] >= query["min_msec"])
        & (df["msec"] <= query["max_msec"])
    )


def get_solution_index(data_df, query: dict):
    filtered_df = data_df.loc[get_valid_indexes(query), :]
    tree = cKDTree(filtered_df.loc[:, pos_cols].values)
    sol_dist, sol_id = tree.query(
        np.array([query["x_position"], query["y_position"], query["z_position"]]),
        workers=-1,
    )
    return filtered_df.iloc[sol_id, :].name


if __name__ == "__main__":
    results = df.loc[
        [get_solution_index(df, query) for query in input_locations], res_cols
    ].to_dict(orient="records")
    Path("output.json").write_text(json.dumps(results))
