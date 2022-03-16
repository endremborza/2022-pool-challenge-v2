import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask
from sklearn.neighbors import KDTree
import numba as nb



app = Flask(__name__)
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
out_cols = ["msec", "subject", "trial"]

pd.read_csv("data.csv").to_pickle("data.pkl")

  
df = pd.read_csv("data.csv")
input_locations = json.loads(Path("C:\Rajk_Prog_II\Prog_datasets\input.json").read_text())

numpy_df = df.loc[:, pos_cols + out_cols].to_numpy()
numpy_input = pd.DataFrame(input_locations).to_numpy()

@app.route("/started")
def started():
    return "numbácsi"

@nb.jit()
def solution(numpy_df, numpy_input):
    out = []
    for i in range(len(numpy_input)):
        mask = (
            (numpy_df[:, [-3]] >= numpy_input[:, [-2]][i][0])
            & (numpy_df[:, [-3]] <= numpy_input[:, [-1]][i][0])
            & (numpy_df[:, [-2]] == numpy_input[:, [-3]][i][0])
        )
        dumma = (
            (numpy_df[:, [0]][mask] - numpy_input[:, [0]][i]) ** 2
            + (numpy_df[:, [1]][mask] - numpy_input[:, [1]][i]) ** 2
            + (numpy_df[:, [2]][mask] - numpy_input[:, [2]][i]) ** 2
        )
        msec = numpy_df[:, [-3]][mask][np.where(dumma == np.amin(dumma))[0][0]]
        subject = numpy_df[:, [-2]][mask][np.where(dumma == np.amin(dumma))[0][0]]
        trial = numpy_df[:, [-1]][mask][np.where(dumma == np.amin(dumma))[0][0]]
        out.append({"msec": msec, "subject": subject, "trial": trial})
    return out

Path("output.json").write_text(json.dumps(solution(numpy_df, numpy_input)))

if __name__ == "__main__":
    app.run(debug=True, port=5999)