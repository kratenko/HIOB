import yaml
import os

os.system("pwd")

filename = "../hiob/conf/datasets/tb100.yaml"

with open(filename, "r") as f:
    tb100 = yaml.safe_load(f)

print(tb100)