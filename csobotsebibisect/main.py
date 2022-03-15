import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.spatial import cKDTree
import pickle
#from constants import tree_path, data_path
from pathlib import Path
import numpy as np
import bisect

if __name__ == "__main__":
    
    pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
    df = pd.read_csv(Path("data.csv")).dropna().loc[:, lambda _df: _df.nunique() != 1]
    df = df.sort_values(by="msec").reset_index()
    input_locations = pd.read_json("input.json")

    out = []
    for minimum, maximum, inputs in zip(
        input_locations["min_msec"],
        input_locations["max_msec"],
        range(len(input_locations)),
    ):
        indmin = bisect.bisect_left(df["msec"], minimum)
        indmax = bisect.bisect_right(df["msec"], maximum)
        data = df.iloc[indmin : int(indmax), :].reset_index()
        data = data[data["subject"] == input_locations["subject"][inputs]]
        tree = cKDTree(data[pos_cols])
        dist, ind = tree.query(input_locations.iloc[int(inputs), 0:3])
        # print (result)
        out.append(data.iloc[ind, :]["index"])

    solutions = df.set_index("index").loc[out, :][["msec", "subject", "trial"]]
    solution_dict = ([solutions.iloc[i].to_dict() for i in range(len(solutions))])
    Path("output.json").write_text(json.dumps(solution_dict))