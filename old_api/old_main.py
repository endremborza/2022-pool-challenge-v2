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

@app.route("/started")
def started():
    return "FING"


@app.route("/")
def solution():
    input_json = json.loads(Path("input.json").read_text())
    result = current_app.tree.query([[r[c] for c in pos_cols] for r in input_json], k=10000)
    indexes = [x[0] for x in result[1]]
    out = [dict(zip(out_cols, x)) for x in current_app.arr[indexes, :]]
    Path("output.json").write_text(json.dumps(out))
    return "FING"


app.dfo = (
    pd.read_csv("data.csv")
    .loc[lambda df: df["entity_id"] == 0, [*pos_cols, *out_cols]]
)

app.coords = np.array(app.dfo[pos_cols])
app.arr = np.array(app.dfo[out_cols])
app.tree = KDTree(app.coords, leaf_size=100)


if __name__ == "__main__":
    app.run(debug=True, port=5200)