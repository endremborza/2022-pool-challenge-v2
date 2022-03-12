from constants import data_path, tree_path
import pickle
import json
from pathlib import Path
import numpy as np

tree = pickle.loads(tree_path.read_bytes())
df = pickle.loads(data_path.read_bytes())
input_locations = json.loads(Path("input.json").read_text())
res_cols = ["msec", "subject", "trial"]


def get_wrong_indexes(query: dict):
    return df[
        (df["subject"] != query["subject"])
        | (df["msec"] < query["min_msec"])
        | (df["msec"] > query["max_msec"])
    ].index.to_list()


def get_solution_index(tree, query: dict):
    wrong_index = get_wrong_indexes(query)
    old_data = tree.data[wrong_index, 0]
    tree.data[wrong_index, 0] = np.nan
    solution = tree.query(
        np.array([query["x_position"], query["y_position"], query["z_position"]])
    )
    tree.data[wrong_index, 0] = old_data
    return solution[1]


if __name__ == "__main__":
    result = df.loc[
        [get_solution_index(tree, query) for query in input_locations], res_cols
    ].to_dict(orient="records")

    Path("output.json").write_text(json.dumps(result))
