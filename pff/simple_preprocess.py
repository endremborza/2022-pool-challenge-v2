import pandas as pd

if __name__ == "__main__":
    pd.read_csv("data.csv").to_pickle("data.pkl")
