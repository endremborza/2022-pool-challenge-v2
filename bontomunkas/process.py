from constants import data_path, tree_path
import pickle
import json
from pathlib import Path
import numpy as np
from modified_tree import query_subset


class TreeSubsets:
    def __init__(self):
        for file in tree_path.iterdir():
            setattr(self, file.stem, pickle.loads(file.read_bytes()))


class DataSubsets:
    def __init__(self):
        for file in data_path.iterdir():
            setattr(self, file.stem, pickle.loads(file.read_bytes()))


def get_range_index(arr, min_value, max_value):
    min_index = arr.searchsorted(min_value, side="left")
    max_index = arr.searchsorted(max_value, side="right")
    return min_index, max_index


pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
res_cols = ["msec", "subject", "trial"]


def get_solution_index(query: dict):
    tree = getattr(tree_collection, query["subject"])
    df = getattr(data_collection, query["subject"])
    min_index, max_index = get_range_index(
        df["msec"].values, query["min_msec"], query["max_msec"]
    )
    sol_id, sol_dist = query_subset(
        tree,
        [query["x_position"], query["y_position"], query["z_position"]],
        np.arange(min_index, max_index),
    )
    return df.loc[sol_id, res_cols].to_dict()


if __name__ == "__main__":
    tree_collection = TreeSubsets()
    data_collection = DataSubsets()
    input_locations = json.loads(Path("input.json").read_text())

    result = [get_solution_index(query) for query in input_locations]
    Path("output.json").write_text(json.dumps(result))
