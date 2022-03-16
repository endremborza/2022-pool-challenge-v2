import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask
from sklearn.neighbors import KDTree
from pathlib import Path
from scipy.spatial import cKDTree
import bisect


app = Flask(__name__)
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
out_cols = ["msec", "subject", "trial"]
input_locations = json.loads(Path("input.json").read_text())


df = pd.read_csv(Path("data.csv")).dropna().loc[:, lambda _df: _df.nunique() != 1]
df = df.sort_values(by="msec").reset_index()

pd.read_csv("data.csv").to_pickle("data.pkl")

@app.route("/started")
def started():
    return "Whoosh"

out = []

@app.route("/")
def solution(minimum, maximum, query):
    for minimum, maximum, query in zip(
        input_locations["min_msec"], input_locations["max_msec"], input_locations.iterrows()
    ):
        indmin = bisect.bisect_left(df["msec"], minimum)
        indmax = bisect.bisect_right(df["msec"], maximum)
        data = df.iloc[indmin : int(indmax), :].reset_index()
        data = data[data["subject"] == query[1]["subject"]]
        pos_arr = np.array([query[1][c] for c in pos_cols], ndmin=2)
        min_ind = ((data[pos_cols] - pos_arr) ** 2).sum(axis=1).idxmin()
        out.append(data.loc[min_ind, :]["index"])
    return "You've been robbed!"

solutions = df.set_index("index").loc[out, :][["msec", "subject", "trial"]]
solution_dict = ([solutions.iloc[i].to_dict() for i in range(len(solutions))])

Path("output.json").write_text(json.dumps(solution_dict))


if __name__ == "__main__":
    app.run(debug=True, port=5501)