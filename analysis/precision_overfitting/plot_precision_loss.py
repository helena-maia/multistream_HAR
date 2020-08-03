import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument("precision_path", action='store', type=str, help="", nargs='+')
parser.add_argument("-labels", action='store', type=str, help="", nargs='+')
parser.add_argument("-loss", action='store_true', help="")
parser.add_argument("-o", action='store', type=str, default="plot.eps", help="")
args = parser.parse_args()

max_epochs = -1
y_all = []

for i, p in enumerate(args.precision_path):
    list_files = sorted(glob.glob(os.path.join(p, "*.txt")))
    ind = 2 if args.loss else 3

    y = np.zeros(len(list_files))
    max_epochs = max(max_epochs, len(list_files))

    for f_path in list_files:
        with open(f_path, "r") as f:
            lines = f.readlines()
            lines = [l for l in lines if 'Epoch:' in l]
            last_line = lines[-1]
            columns = last_line.split("\t")

            epoch = columns[0].split("[")[1]
            epoch = epoch.split("]")[0]

            column = columns[ind].replace("\n","")
            avg = column.split("(")[1]
            avg = float(avg.split(")")[0])
            
            y[int(epoch)] = avg

    y_all.append(y)

if args.loss:
    top = max([y.max() for y in y_all]) #max(y_all)+1
    step = 1.
    loc = "upper right"
else:
    top = 100.+1
    step = 10.
    loc = "lower right"

colors = plt.get_cmap('Set1').colors
plots = []

plt.ylim(bottom=0., top=top)
plt.xlim(left=0, right=max_epochs+1)
plt.hlines(np.arange(0,top,step), xmin=0, xmax=max_epochs+1, linestyles='dashed', color='lightgray')

plt.ylabel('Precision', fontsize=12)
plt.xlabel('Number of epochs', fontsize=12)
plt.xticks(np.arange(0, max_epochs+1, 25))

for i, y in enumerate(y_all):
    X = np.arange(len(y))
    plots.append(plt.plot(X, y, color=colors[i])[0])

plt.legend(plots, args.labels, loc=loc, title="Modality", ncol=1)
fig = plt.gcf()
#fig.tight_layout()
fig.set_size_inches(10, 5)
fig.savefig(args.o, dpi=100, bbox_inches='tight')

