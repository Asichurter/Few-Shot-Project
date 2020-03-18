import requests
import os

port = 1337
name = "Backdoor.Win32.Bifrose.aaex"

REST_URL = "http://localhost:{port}/tasks/create/file".format(port=port)
SAMPLE_FILE = "D:/peimages/PEs/cluster/train/Backdoor.Win32.Bifrose/{filename}".format(filename=name)
HEADERS = {"Authorization": "Bearer IC-BAaVpgLndSxozRb7XsQ"}

with open(SAMPLE_FILE, "rb") as sample:
    files = {"file": (name, sample)}
    r = requests.post(REST_URL, headers=HEADERS, files=files)

# Add your code to error checking for r.status_code.

task_id = r.json()["task_id"]

# Add your code for error checking if task_id is None.