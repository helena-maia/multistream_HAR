import argparse
import itertools
import operator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='')
parser.add_argument('list', type=str, help='path to list', nargs='+')
parser.add_argument('-o', default="histogram.eps", type=str,
                        help='path to save the Histogram (default: histogram.eps)')

def get_info(path_file):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    duration = []
    classes = []

    for line in lines_file: 
        line_info = line.split(' ')
        duration.append(int(line_info[1]))
        classes.append(int(line_info[2]))

    return np.array(duration), np.array(classes)

def plot(path_file, output):
    if not isinstance(path_file, list): path_file = list(path_file)
    avg_hist = None
    n_files = 0

    for p in path_file:
        _, classes = get_info(p)
        n_classes = len(np.unique(classes))
        n_samples = float(len(classes))
        count = np.array([len(classes[classes==i]) for i in range(n_classes)], dtype=float)

        if avg_hist is None: avg_hist = count
        else: avg_hist += count
        n_files += 1

    avg_hist /= n_files
    max_nsamples = avg_hist.max()+2

    print(avg_hist.min(), avg_hist.max(), avg_hist.sum())

    colors = plt.get_cmap('Dark2').colors
    plt.hlines(avg_hist.mean(), xmin=-5, xmax=105, linestyles='solid', color=colors[1])
    plt.hlines(np.arange(20, max_nsamples, 20), xmin=-5, xmax=105, linestyles='dashed', color='lightgray', zorder=-1)
    plt.xticks(np.arange(0, 100, 10))
    yticks = np.arange(0, max_nsamples, 20)
    yticks = np.append(yticks, avg_hist.mean())
    plt.yticks(yticks)
    plt.xlim(-5,105)
    plt.ylim(0,max_nsamples)
    plt.xlabel("Classes", fontsize=12)
    plt.ylabel("Number of samples", fontsize=12)
    plt.bar(np.arange(len(avg_hist)), avg_hist, color=colors[0])
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    fig.tight_layout()
    fig.savefig(output, dpi=100)


def main():
    args = parser.parse_args()
    path, _ = os.path.split(args.o)

    if not os.path.isdir(path) and path != "":
        print("Creating output directory: ", path)
        os.makedirs(path)

    plot(args.list, args.o)


if __name__ == '__main__':
    main()    
