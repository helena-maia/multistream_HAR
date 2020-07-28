import argparse
import itertools
import operator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='')
parser.add_argument('dataset_list', type=str, help='path to dataset list')
parser.add_argument('-o', default="histogram.eps", type=str,
                        help='path to save the Histogram (default: histogram.eps)')
parser.add_argument('-g', action='store_true',
                        help='general')

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


def stacked_hist(duration, classes, edges):
    stack = []
    n_classes = len(np.unique(classes))
    count = np.array([len(classes[classes==i]) for i in range(n_classes)], dtype=float)

    for end in edges[1:]:
        filtered = classes[duration<=end]
        hist = np.array([len(filtered[filtered==i]) for i in range(n_classes)], dtype=float)
        hist = (hist / count)*100.
        stack.append(hist)

    return np.array(stack)


def plot(path_file, output, chart_type="g"):
    duration, classes = get_info(path_file)

    edges = [0,51,101,301,501,duration.max()]
    colors = plt.get_cmap('Dark2').colors
    ticks = list(range(len(edges)-1))
    labels = ["[{},{}]".format(edges[i],edges[i+1]-1) for i in ticks]

    if chart_type == "g":
        hist, edges = np.histogram(duration, bins=edges)
        hist = (hist*100.)/len(duration)
        for y in range(20,100,20): plt.axhline(y=y, linestyle='--', color='gray',zorder=-1)
        plt.xticks(ticks, labels, fontsize=9)
        plt.ylim(0,100)
        plt.xlabel("Length", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.bar(np.arange(len(hist)), hist, color=colors)
        fig = plt.gcf()
        fig.set_size_inches(5, 5)
        fig.savefig(output, dpi=100)

        print(hist)
    else:
        # per class
        stack = stacked_hist(duration, classes, edges)
        p = [plt.bar(np.arange(len(stack[0])), stack[0], color=colors[0])[0]]
        labels = ["[{},{}]".format(edges[0],edges[1]-1)]
        for i, hist in enumerate(stack[1:]):
            diff = hist - stack[i]
            p.append(plt.bar(np.arange(len(hist)), diff, bottom=stack[i], color=colors[i+1])[0])
            labels.append("[{},{}]".format(edges[i+1],edges[i+2]-1))
	
        plt.legend(p, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), title="Length", ncol=len(edges))
        plt.ylim(0,100)
        n_classes = len(np.unique(classes))
        plt.xticks(np.arange(0,n_classes,10))
        plt.xlabel("Classes", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        fig = plt.gcf()
        fig.set_size_inches(10, 5)
        fig.savefig(output, dpi=100)




def main():
    args = parser.parse_args()
    path, _ = os.path.split(args.o)

    if not os.path.isdir(path) and path != "":
        print("Creating output directory: ", path)
        os.makedirs(path)

    plot(args.dataset_list, args.o, "g" if args.g else "e")


if __name__ == '__main__':
    main()    
