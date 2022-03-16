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
    res_cols = ["msec", "subject", "trial"]

    input_locations = pd.read_json("input.json")
    df = pickle.loads(data_path.read_bytes())
    df = df.sort_values(by="msec").reset_index()

    out = []
    for minimum, maximum, query in zip(
        input_locations["min_msec"], input_locations["max_msec"], input_locations.iterrows()
    ):
        indmin = bisect.bisect_left(df["msec"], minimum)
        indmax = bisect.bisect_right(df["msec"], maximum)
        data = df.iloc[indmin : int(indmax), :].reset_index()
        data = data[data["subject"] == query[1]["subject"]]
        # (data[pos_cols])
        pos_arr = np.array([query[1][c] for c in pos_cols], ndmin=2)
        min_ind = ((data[pos_cols] - pos_arr) ** 2).sum(axis=1).idxmin()
        out.append(data.loc[min_ind, :]["index"])


    solutions = df.set_index("index").loc[out, :][["msec", "subject", "trial"]]
    solution_dict = ([solutions.iloc[i].to_dict() for i in range(len(solutions))])

    Path("output.json").write_text(json.dumps(solution_dict))