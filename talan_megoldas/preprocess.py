import pandas as pd
import subprocess
import requests
import time

if __name__ == "__main__":
    pd.read_csv("data.csv").to_pickle("data.pkl")

proc = subprocess.Popen(["python", "main.py"])

while True:
    try:
        time.sleep(1)
        resp = requests.get("http://localhost:5555/started")
        time.sleep(4)
        break
    except Exception as e:
        pass