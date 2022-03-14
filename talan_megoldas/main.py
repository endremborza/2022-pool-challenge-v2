import json
import pandas as pd
import numpy as np
from pathlib import Path
from flask import Flask
from flask import current_app
from sklearn.neighbors import KDTree
import numba as nb

#creating essentials

app = Flask(__name__)
pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
out_cols = ["msec", "subject", "trial"]

#creating home page

@app.route("/home")
def home():
    return "Hello World!"

#creating relevant page with the solution

@app.route("/") #creates page in which the smallest distance is calculated
@nb.jit(nopython=True, parallel=True) #speeds up functions. 
                                      #nopython=True: does not compile code in python if able. 
                                      #parallel=True: speeds up process when paralell processes would otherwise slow it down
def solution():
    input_json = json.loads(Path("input.json").read_text())
    df = pd.read_pickle("data.pkl") #preprocesben át kell alakítani pickle-é, az Endre intro feladatából átemelt rész, a for loopban használja. 
                                    #Ez szerintem a lent definiált app.dfo miatt megoldható könnyebben is. Utána kell még nézni
    for query in input_json: #for ciklus helyett map?
        df.drop_duplicates(subset=out_cols).drop_duplicates(subset=pos_cols) #ide átkerült a duplikátumok kidobása. 
                                                                             #2 út van: 1) itt kidobjuk a duplikátumokat / 2) k=5000-el (vagy nagyobbal kell lent menni)
                                                                             #sejtésem szerint ez lesz a gyorsabb út.

        #this can be better 
        df[(df["subject"] == query["subject"]) & (df["msec"] >= query["min_msec"]) & (df["msec"] <= query["max_msec"])]
        #this can be better 

        #this is good
        result = current_app.tree.query([[r[c] for c in pos_cols] for r in input_json]) #ha a for loop után kidobni mindig a duplikátumokat túl lassú lenne ezt kell használni:
                                                                                        #result = current_app.tree.query([[r[c] for c in pos_cols] for r in input_json], k = 5000)
        #this is good
        
    #Endre solution - does not minimize (adatkidobás miatt nem kellett itt már)
    #1) útnál ez alkalmazandó
        indexes = [x[0] for x in result[1]]
        out = [dict(zip(out_cols, x)) for x in current_app.arr[indexes, :]]
        Path("output.json").write_text(json.dumps(out))

    #Sebitree solution - does minimize, uses other format for kdtree
    #2) útnál ez alkalmazandó
        #dist, ind = tree.query(input_locations, k = 100)
        #results_index = [min([i for d, i in zip(d_vec, i_vec) if d == d_vec.min()]) for d_vec, i_vec in zip(dist, ind)]
        #results = df.iloc[results_index, :][out_cols].to_dict("records")
        #Path("output.json").write_text(json.dumps(results))

    return "Goodbye World!"


#creating elements for api usage

#innen kikerült az azonos koordinátájú pontok törlése, átment a for loopba
app.dfo = (
    pd.read_csv("data.csv")
    .loc[lambda df: df["entity_id"] == 0, [*pos_cols, *out_cols]]
    .dropna(how="any")
)

app.coords = np.array(app.dfo[pos_cols])
app.arr = np.array(app.dfo[out_cols])
app.tree = KDTree(app.coords, leaf_size=50)

#starting app

if __name__ == "__main__":
    app.run(debug=True, port=5112)