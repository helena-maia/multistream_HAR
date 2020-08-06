import argparse
import glob
import os
import json
import numpy as np
from early_stopping.pytorchtools import EarlyStopping

parser = argparse.ArgumentParser(description='')
parser.add_argument("es_json", action='store', type=str, help="")
parser.add_argument("precision_path", action='store', type=str, help="")
parser.add_argument("delta", action='store', type=float, help="")
parser.add_argument("patience", action='store', type=int, help="")
args = parser.parse_args()

es = EarlyStopping(verbose = False, patience = args.patience, delta = args.delta)
best_epoch = -1

with open(args.es_json, 'r') as json_file:
    es_dict = json.load(json_file)

    epochs = sorted(es_dict.keys())[:-1]

    for epoch in epochs:
            e = int(epoch)
            val_loss = es_dict[epoch]['val_loss']

            is_best = es(val_loss, e)
            print(epoch, val_loss, is_best, es.best_score)

            if is_best:
                    best_epoch = e + 1

            if es.early_stop:
                break

if not es.early_stop:
    print(-1, -1, -1) 
else:
    prec_path = os.path.join(args.precision_path, "{:03d}*.txt".format(best_epoch))
    prec_path = glob.glob(prec_path)

    if len(prec_path) != 1:
        print(-1, -1, -1)

    else:
        with open(prec_path[0], "r") as prec_file:
            lines_ = prec_file.readlines()

            lines = [l for l in lines_ if 'Epoch:' in l]
            last_line = lines[-1]
            columns = last_line.split("\t")
            column_p = columns[3].replace("\n","")
            train_prec = column_p.split("(")[1]
            train_prec = float(train_prec.split(")")[0])

            lines = [l for l in lines_ if 'Validation:' in l]
            last_line = lines[-1]
            columns = last_line.split("\t")
            column_p = columns[3].replace("\n","")
            val_prec = column_p.split("(")[1]
            val_prec = float(val_prec.split(")")[0])

            print(best_epoch, train_prec, val_prec)
