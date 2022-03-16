import pickle
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    
    pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
    
    df = (
        (
            pd.read_csv(Path("data.csv"))
            .drop_duplicates(pos_cols)
            .dropna()
            .loc[:, lambda _df: _df.nunique() != 1]
        )
        .sort_values(by="msec")
        .reset_index()
    )
    
    df.to_pickle("data.pkl")