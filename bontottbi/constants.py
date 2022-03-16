from pathlib import Path

data_path = Path('data')

data_path.mkdir(exist_ok=True)

pos_cols = [f"{ax}_position" for ax in ["x", "y", "z"]]
res_cols = ["msec", "subject", "trial"]