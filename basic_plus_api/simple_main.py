import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask
from flask import current_app
from sklearn.neighbors import KDTree

app = Flask(__name__)
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
out_cols = ["msec", "subject", "trial"]

pd.read_csv("data.csv").to_pickle("data.pkl")

@app.route("/started")
def started():
    return "FING"


@app.route("/")
def solution():
    df = pd.read_pickle("data.pkl")
    input_locations = json.loads(Path("input.json").read_text())
    out = []
    for query in input_locations:
        filt = (
            (df["subject"] == query["subject"])
            & (df["msec"] >= query["min_msec"])
            & (df["msec"] <= query["max_msec"])
        )
        pos_arr = np.array([query[c] for c in pos_cols], ndmin=2)
        min_ind = ((df.loc[filt, pos_cols] - pos_arr) ** 2).sum(axis=1).idxmin()
        out.append(df.loc[min_ind, out_cols].to_dict())

    Path("output.json").write_text(json.dumps(out))
    return "FING"

app.dfo = (
    pd.read_csv("data.csv")
    .loc[lambda df: df["entity_id"] == 0, [*pos_cols, *out_cols]]
    .dropna(how="any")
)

app.coords = np.array(app.dfo[pos_cols])
app.arr = np.array(app.dfo[out_cols])
app.tree = KDTree(app.coords, leaf_size=50)


if __name__ == "__main__":
    app.run(debug=True, port=5120)