import subprocess
import requests
import time

proc = subprocess.Popen(["python", "simple_main.py"])

while True:
    try:
        time.sleep(1)
        resp = requests.get("http://localhost:5120/started")
        time.sleep(4)
        break
    except Exception as e:
        pass