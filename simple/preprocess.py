import pandas as pd
import pickle
from pathlib import Path


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    df_dic = {gid: gdf.sort_values("msec") for gid, gdf in df.groupby("subject")}
    Path("data.pkl").write_bytes(pickle.dumps(df_dic))
