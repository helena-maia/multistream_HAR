import glob
import json
import os
import numpy as np

npy_list = glob.glob("../LVR/3.vr_stream/test/NPYS_overfitting/*")
header = []
summary = []

for npy_item in npy_list:
    args = os.path.join(npy_item, "args.json")
    npy_path = os.path.join(npy_item, "*.npy")

    args_dict = {}
    with open(args) as f:
        args_dict = json.load(f)

    npy_files = glob.glob(npy_path)
    npy = "none" if len(npy_files) == 0 else npy_files[0] if len(npy_files) == 1 else "multiples"

    if header == []:
        header = ["timestamp"]
        header += sorted(args_dict.keys())
        header += ["npy_file"]
        summary += [header]

    timestamp = npy_item.split("/")[-1]
    data = [args_dict.get(k,"ND") for k in header[1:-1]]
    data = [timestamp] + data + [npy]

    summary += [data]

summary = np.savetxt("npy_overfitting_lvr_summary.csv", summary, fmt="%s", delimiter="\t")




