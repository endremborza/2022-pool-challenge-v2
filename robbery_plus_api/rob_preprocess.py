import subprocess
import requests
import time

proc = subprocess.Popen(["python", "adv_main.py"])

while True:
    try:
        resp = requests.get("http://localhost:5500/started")
        time.sleep(1)
        break
    except Exception as e:
        pass