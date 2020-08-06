import glob
import json
import os
import numpy as np

log_list = glob.glob("../LVR/3.vr_stream/log_es_lvr/*")
header = []
summary = []

for ll in log_list:
    args = os.path.join(ll, "args.json")
    early_stop = os.path.join(ll, "early_stopping.json")

    args_dict, early_dict = {}, {}


    with open(args) as f:
        args_dict = json.load(f)
        if 'es' not in args_dict:
            args_dict["es"] = True

    try:
        with open(early_stop) as f:
            early_dict = json.load(f)
    except:
        early_dict = {}
        

    if header == []:
        header = ["timestamp"]
        header += sorted(args_dict.keys())
        header += ["best_epoch", "best_val"]
        summary += [header]

    timestamp = ll.split("/")[-1]
    data = [args_dict.get(k, None) for k in header[1:-2]]

    try:
        best_epoch = sorted([k for k in early_dict.keys() if k != "config"])[-8]
    except:
        data = [timestamp] + data + [-1, -1]
    else:
        best_data = early_dict[best_epoch]
        data = [timestamp] + data + [best_epoch, best_data["val_loss"]]

    summary += [data]

summary = np.savetxt("log_es_lvr_summary.csv", summary, fmt="%s", delimiter="\t")
