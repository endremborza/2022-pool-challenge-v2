import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask
from flask import current_app
from sklearn.neighbors import KDTree

app = Flask(__name__)

pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
res_cols = ["msec", "subject", "trial"]
cat_col = "subject"

class PoolTree:

    tree_keys = pos_cols + res_cols[:1]
    return_keys = res_cols

    def __init__(self, base_df: pd.DataFrame):

        self.coords = np.array(base_df[self.tree_keys])
        self.res_df = base_df[self.return_keys]
        self.tree = KDTree(self.coords, leaf_size=50)

    def query_w_filter(self, input_list):

        input_points = []
        input_filters = []
        for indic in input_list:
            indic["msec"] = (indic["min_msec"] + indic["max_msec"]) / 2
            input_filters.append([*([-1] * len(pos_cols)), indic["max_msec"] - indic["msec"]])
            input_points.append([indic[k] for k in self.tree_keys])

        k_dist_list, k_ind_list = self.tree.query_filtered(
            np.array(input_points), np.array(input_filters), k = 5
        )
        inds = []
        for i, (k_dists, k_inds) in enumerate(zip(k_dist_list, k_ind_list)):
            inds.append(self.res_df.index[k_inds[k_dists == k_dists.min()]].min())
        return self.res_df.loc[inds, :].to_dict("records")


@app.route("/started")
def started():
    return "FING"


@app.route("/")
def solution():
    input_json = json.loads(Path("input.json").read_text())
    cat_groups = defaultdict(list)
    cat_indices = defaultdict(list)
    for i, in_dic in enumerate(input_json):
        sr = in_dic[cat_col]
        cat_groups[sr].append(in_dic)
        cat_indices[sr].append(i)

    out = {}
    for sr, sinp in cat_groups.items():
        spec_tree = current_app.star_dict.get(sr)
        s_out = spec_tree.query_w_filter(sinp)
        for o, ind in zip(s_out, cat_indices[sr]):
            out[ind] = o

    out_list = [v for k, v in sorted(out.items(), key=lambda kv: kv[0])]
    Path("output.json").write_text(json.dumps(out_list))
    return "FING"


app.dfo = pd.read_csv("data.csv").loc[:, res_cols + pos_cols]
app.star_dict = {}

for _gid, _gdf in app.dfo.groupby(cat_col):
    app.star_dict[_gid] = PoolTree(_gdf)

if __name__ == "__main__":
    app.run(debug=True, port=5112, threaded=True)
