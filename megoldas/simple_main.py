import json
import pandas as pd
import numpy as np
from pathlib import Path
import scipy.spatial.distance

pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
out_cols = ["msec", "subject", "trial"]

if __name__ == "__main__":

    df = pd.read_pickle("data.pkl")
    input_locations = json.loads(Path("input.json").read_text())
    result_locations = json.loads(Path("results.json").read_text())
    df.drop_duplicates(subset=out_cols).drop_duplicates(subset=pos_cols)
    input = np.array([list(i.values())[:3] for i in input_locations])
    lista = []
    for input_row in input_locations:
        df_ = df[
            (df["subject"] == input_row["subject"])
            & (df["msec"] >= input_row["min_msec"])
            & (df["msec"] <= input_row["max_msec"])
        ]
        input = list(input_row.values())[:3]
        df_t = df_.reset_index().loc[:, ["x_position", "y_position", "z_position"]]
        d = scipy.spatial.distance.cdist(list(df_t.values), np.array([input]))
        shortest_index = d.argmin(axis=0)
        lista.append(df_.iloc[shortest_index[0], [0, -2, -1]].to_dict())
    Path("output.json").write_text(json.dumps(lista))
