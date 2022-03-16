"""
Packages
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
import bisect
from sklearn.neighbors import KDTree


if __name__ == "__main__":

    """
    Read Data and Input
    """
    df = pd.read_pickle("data.pkl")
    input_locations = pd.read_json("input.json")
    
    
    """
    Go crazy
    """
    pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
    res_cols = ["msec", "subject", "trial"]
    out=[]
    
    for minimum, maximum, query in zip(
        input_locations["min_msec"], input_locations["max_msec"], input_locations.iterrows()):
        
        """
        1. Meghatározzuk a sorba rendezett adatban a minimum és maximum indexeket.
        2. Leszűrjük az adatot, hogy a minimum és maximum index közötti sorok maradjanak meg 
        és az eredeti adat indexéből oszlopot csinálunk a szűrt adatban.
        3. Leszűrjük az adatot a megfelelő alanyra
        """
        indmin = bisect.bisect_left(df["msec"], minimum)
        indmax = bisect.bisect_right(df["msec"], maximum) 
        data = df.iloc[indmin : int(indmax), :].reset_index()
        data = data[data["subject"] == query[1]["subject"]]
        
        """
        Csinálok egy position arrayt, amiben az x, y, z koordinátái vannak benne az input lokációnak
        """
        pos_arr = np.array([query[1][c] for c in pos_cols], ndmin=2)
        
        """
        Ha nagyobb a szűrt adat 100 sornál, akkor fában keresek, különben lineárisan
        """
        if len(data)>100:
            """
            KDTree:
            1. Építünk egy fát a szűrt adatban
            2. Megkeressük a fában a legközelebbi szomszédját az inputnak
            """
            tree = KDTree(data.loc[:, pos_cols])
            dist, ind = tree.query(pos_arr, k = 1)
            min_ind=ind[0][0]
            out.append(data.iloc[min_ind, :]["index"])    
        else:
            """
            A megszűrt adatnak veszem az x, y, z oszlopait és kiszámolom az euklideszi távolságokat
            az input lokációtól és megtartom a legkisebbet ezek közül
            """
            min_ind = ((data[pos_cols] - pos_arr) ** 2).sum(axis=1).idxmin()
            out.append(data.loc[min_ind, :]["index"]) 
    
    solutions = df.set_index("index").loc[out, :][["msec", "subject", "trial"]]
    solution_dict = ([solutions.iloc[i].to_dict() for i in range(len(solutions))])
    Path("output.json").write_text(json.dumps(solution_dict))