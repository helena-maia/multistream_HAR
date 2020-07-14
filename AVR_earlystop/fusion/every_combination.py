from fusion import fusion, methods
import argparse
import itertools
import json
import os

json_file = open("../utils/config.json", "r")
npy_dict = json.load(json_file)

def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-s', default=[1,2,3], type=int, nargs='+', metavar='S',
                        help='which split of data to work on (default: 1 2 3)')
    parser.add_argument('-d', nargs='+', default=["ucf101", "hmdb51"], type=str, metavar='DATASET',
                        help='datasets (default: ucf101 hmdb51)')
    parser.add_argument('-o', default="fusion.csv", type=str, metavar='OUTPUT',
                        help='output path (default: fusion.csv)')
    parser.add_argument('-t', default="no_retrain", type=str, metavar='TYPE',
                        help='experiment group: ' + ",".join(npy_dict.keys()),
                        choices = npy_dict.keys())
    parser.add_argument('--modalities', type=str, nargs='+', default=["rgb","flow"],
                        help='modalities (list) (default: rgb flow')
    parser.add_argument('--settings', metavar='DIR', default='../datasets/settings_earlystop',
                        help='path to dataset setting files')

    return parser.parse_args()

args = get_args()

train_type = args.t
dataset = args.d
npy_path_fmt = "../scripts/NPYS/{}/{}_{}_inception_v3_s{}.npy"
splits = args.s
settings = args.settings
modalities = args.modalities
output = args.o
#methods.remove("individual")
#methods.remove("SVM")
#methods.remove("FC")
#methods.remove("sugeno_fuzzy")
methods = ["sugeno_fuzzy"]

npy_dict = npy_dict[train_type]

n_modalities = len(modalities)
comb = None
for r in range(2, n_modalities+1): # combinations of 2 or more modalities
    c = itertools.combinations(modalities, r=r)
    comb = itertools.chain(comb, c) if comb else c

it = itertools.product(dataset, splits, methods, comb)

with open(output, "w") as f:
    f.write("dataset\tsplit\tmethod\tcombination\tparam\tprec\n")

for d, s, m, c in it:
    key = "{}_s{}".format(d,s)

    npy_paths = []
    for mod in c:
        npy_path = npy_path_fmt.format(npy_dict[key][mod],
                                       d, mod, s)
        if not os.path.isfile(npy_path):
           npy_paths = []
           break

        npy_paths.append(npy_path)

    if len(npy_paths) >= 2:
        args2 = argparse.Namespace(d=d, m=m, npy_paths=npy_paths, 
                                  s=s, settings=settings)

        print(args2)
        ret = fusion(args2)

        if ret:
            param, _, prec = ret
            with open(output, "a") as f:
                f.write("{}\t{}\t{}\t{}\t{}\t{:.04f}\n".format(d, s, m, c, param, prec))
