from constants import data_path, tree_path
import pickle
import json
from pathlib import Path
import numpy as np
from modified_tree import query_subset

tree = pickle.loads(tree_path.read_bytes())
df = pickle.loads(data_path.read_bytes())
input_locations = json.loads(Path("input.json").read_text())
res_cols = ["msec", "subject", "trial"]

def get_wrong_indexes(query: dict):
    return df[
        lambda _df: (
            (_df["subject"] == query["subject"])
            & (_df["msec"] >= query["min_msec"])
            & (_df["msec"] <= query["max_msec"])
        )
    ].index.values

def get_solution_index(tree, query: dict):
    sol_id, sol_dist = query_subset(
        tree,
        [query["x_position"], query["y_position"], query["z_position"]],
        get_wrong_indexes(query),
    )
    return sol_id


if __name__ == "__main__":
    result = df.loc[
        [get_solution_index(tree, query) for query in input_locations], res_cols
    ].to_dict(orient="records")

    Path("output.json").write_text(json.dumps(result))
    # out = json.loads(Path("results.json").read_text())
    # assert out == result 