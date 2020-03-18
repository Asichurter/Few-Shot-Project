import json
import pprint

path = 'C:/Users/Asichurter/Desktop/report.json'

with open(path, 'r') as f:
    r = json.load(f)
    print(r['target']['file']['name'])