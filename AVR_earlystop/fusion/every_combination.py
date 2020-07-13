from fusion import fusion, methods
import argparse
import itertools
import json

train_type = "no_retrain" # or "retrain"

json_file = open("../utils/config.json", "r")
npy_dict = json.load(json_file)[train_type]

dataset = ['ucf101','hmdb51']
npy_path_fmt = "../scripts/NPYS/{}/{}_{}_inception_v3_s{}.npy"
splits = [1, 2, 3]
settings = "../datasets/settings_earlystop"
modalities = ["rgb", "flow"] # choices: ["rhythm", "rgb", "flow"]
output = "output.csv"

n_modalities = len(modalities)
comb = None
for r in range(2, n_modalities+1): # combinations of 2 or more modalities
    c = itertools.combinations(modalities, r=r)
    comb = itertools.chain(comb, c) if comb else c

it = itertools.product(dataset, splits, methods, comb)

with open(output, "w") as f:
    f.write("dataset \t split \t method \t combination \t prec\n")

for d, s, m, c in it:
    key = "{}_s{}".format(d,s)
    
    npy_paths = []
    for mod in c:
        npy_path = npy_path_fmt.format(checkpoint_dict[key][mod],
                                       d, mod, s)
        npy_paths.append(npy_path)

    args = argparse.Namespace(d=d, m=m, npy_paths=npy_paths, 
                              s=s, settings=settings)

    print(args)
    prec = fusion(args)
    
    with open(output, "a") as f:
        f.write("{}\t{}\t{}\t{}\t{:04f}\n".format(d, s, m, c, prec))