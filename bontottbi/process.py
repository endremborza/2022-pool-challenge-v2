from constants import data_path, pos_cols, res_cols
import pickle
import json
from pathlib import Path
import numpy as np

class DataSubsets:
    def __init__(self):
        for file in data_path.iterdir():
            setattr(self, file.stem, pickle.loads(file.read_bytes()))


def get_range_index(arr, min_value, max_value):
    min_index = arr.searchsorted(min_value, side="left")
    max_index = arr.searchsorted(max_value, side="right")
    return min_index, max_index


def get_solution_index(query: dict):
    df = getattr(data_collection, query["subject"])
    min_index, max_index = get_range_index(
        df["msec"].values, query["min_msec"], query["max_msec"]
    )
    data = df.iloc[min_index:max_index, :][pos_cols]
    pos_arr = np.array([query["x_position"], query["y_position"], query["z_position"]], ndmin=2)
    min_ind = ((data - pos_arr) ** 2).sum(axis=1).idxmin()
    return df.loc[min_ind, res_cols].to_dict()


if __name__ == "__main__":
    data_collection = DataSubsets()
    input_locations = json.loads(Path("input.json").read_text())

    result = [get_solution_index(query) for query in input_locations]
    Path("output.json").write_text(json.dumps(result))
