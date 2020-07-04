import glob
import json
import os

log_list = glob.glob("log/*")
header = []

for ll in log_list:
    args = os.path.join(ll, "args.json")
    early_stop = os.path.join(ll, "early_stopping.json")

    args_dict, early_dict = {}, {}


    with open(args) as f:
        args_dict = json.load(f)

    with open(early_stop) as f:
        early_dict = json.load(f)

    if header == []:
        header = sorted(args_dict.keys())
        header+= ["best_epoch", "best_val"]

    #print(args_dict.keys())
    #print(early_dict.keys())

print(header)





