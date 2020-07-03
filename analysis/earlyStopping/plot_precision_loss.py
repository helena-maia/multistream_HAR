import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument("precision_path", action='store', type=str, help="")
parser.add_argument("-l", action='store_true', help="")
args = parser.parse_args()

list_files = sorted(glob.glob(os.path.join(args.precision_path, "precision", "*.txt")))
ind = 2 if args.l else 3

X = []
y = []

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
		
		X.append(epoch)
		y.append(avg)

if args.l:
    top = max(y)+1
    step = 1.
else:
    top = 100.+1
    step = 10.

plt.ylim(bottom=0., top=top)
plt.xlim(left=-1, right=len(X)+1)
plt.hlines(np.arange(0,top,step), xmin=0, xmax=len(X), linestyles='dashed', color='lightgray')
plt.vlines([len(X)-7], ymin=0, ymax=top, linestyles='dashed', color='red')

plt.xticks(np.arange(0, len(X), 5))
plt.plot(X, y)
plt.show()