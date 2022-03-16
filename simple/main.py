import numpy as np
import pickle
import json
from pathlib import Path

pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
res_cols = ["msec", "subject", "trial"]


if __name__ == "__main__":
    df_dic = pickle.loads(Path("data.pkl").read_bytes())
    input_locations = json.loads(Path("input.json").read_text())
    out = []
    for i, query in enumerate(input_locations):
        _df = df_dic[query["subject"]]
        min_ind = np.searchsorted(_df["msec"].values, query["min_msec"], side="left")
        max_ind = np.searchsorted(_df["msec"].values, query["max_msec"], side="right")
        pos_arr = np.array([query[c] for c in pos_cols], ndmin=2)
        res_ind = _df.iloc[min_ind:max_ind, :].pipe(lambda df: ((df[pos_cols] - pos_arr) ** 2).sum(axis=1)).idxmin()
        out.append(_df.loc[res_ind, res_cols].to_dict())

    Path("output.json").write_text(json.dumps(out))
