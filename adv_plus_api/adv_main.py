import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask
from sklearn.neighbors import KDTree


app = Flask(__name__)
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
out_cols = ["msec", "subject", "trial"]

df = pd.read_csv("data.csv")
input_locations = json.loads(Path("C:\Rajk_Prog_II\Prog_datasets\input.json").read_text())


@app.route("/started")
def started():
    return "FING"
    
numpy_df = df.loc[:, pos_cols + out_cols].to_numpy()
numpy_input = pd.DataFrame(input_locations).to_numpy()


@app.route("/")
def solution(numpy_df, numpy_input):
    out = []
    for i in range(len(numpy_input)):
        mask = (
                (numpy_df[:, [-3]] >= numpy_input[:, [-2]][i][0])
                & (numpy_df[:, [-3]] <= numpy_input[:, [-1]][i][0])
                & (numpy_df[:, [-2]] == numpy_input[:, [-3]][i][0])
            )
        merge_pos_arr = np.column_stack([numpy_df[:,[0]][mask], numpy_df[:,[1]][mask], numpy_df[:,[2]][mask]])
        tree = KDTree(merge_pos_arr, leaf_size=50)
        dist, ind = tree.query(numpy_input[i][:3].reshape(1,-1), k = 1)
        msec = numpy_df[:, [-3]][mask][ind][0][0]
        subject = numpy_df[:, [-2]][mask][ind][0][0]
        trial = numpy_df[:, [-1]][mask][ind][0][0]
        out.append({"msec": msec, "subject": subject, "trial": trial})
    return out
    
Path("output.json").write_text(json.dumps(solution(numpy_df, numpy_input)))

app.dfo = (
    pd.read_csv("data.csv")
    .loc[lambda df: df["entity_id"] == 0, [*pos_cols, *out_cols]]
    .dropna(how="any")
)

app.coords = np.array(app.dfo[pos_cols])
app.arr = np.array(app.dfo[out_cols])
app.tree = KDTree(app.coords, leaf_size=50)


if __name__ == "__main__":
    app.run(debug=True, port=5500)