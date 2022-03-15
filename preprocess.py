import pandas as pd
import pickle

data = pd.read_csv("data.csv")

data["msec"] = data["msec"].astype(np.int32)
data["subject"] = data["subject"].astype("category")

cols_1 = ['msec', 'x_position', 'y_position', 'z_position', 'subject','trial']

data = data.loc[:,cols_1]

data.to_pickle("data.pkl")
