import argparse
import glob
import os
import numpy as np
import json
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument("es_log", action='store', type=str, help="json file")
args = parser.parse_args()

with open(args.es_log, "r") as json_file:
    es_dict = json.load(json_file)
    epochs = sorted([k for k in es_dict.keys() if k != "config"])
    X = []
    val_loss_list = []

    for e in epochs:
        X.append(float(e))
        val_loss_list.append(es_dict[e]['val_loss'])

    top = max(val_loss_list)+1.
    step = 1.

    plt.ylim(bottom=0., top=top)
    plt.xlim(left=-1, right=len(X)+1)
    plt.hlines(np.arange(0,top,step), xmin=0, xmax=len(X), linestyles='dashed', color='lightgray')
    plt.vlines([len(X)-7], ymin=0, ymax=top, linestyles='dashed', color='red')

    plt.xticks(np.arange(0, len(X), 5))
    plt.plot(X, val_loss_list)
    plt.show()
