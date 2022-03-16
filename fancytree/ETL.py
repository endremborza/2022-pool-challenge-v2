import subprocess
import os
import requests
import time

proc = subprocess.Popen(["python", "flask_prep.py"])

while True:
    try:
        time.sleep(1)
        requests.get("http://127.0.0.1:5112/started")
        time.sleep(4)
        break
    except Exception as e:
        pass
