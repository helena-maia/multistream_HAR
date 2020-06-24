import argparse
import itertools
import operator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='')
parser.add_argument('npy_dir', type=str, help='path to npy files')
parser.add_argument('splits_dir', type=str, help='path to split files')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
          help='which split of data to work on (default: 1)')
parser.add_argument('-a', default="inception_v3", type=str,
                        help='architecture (default: inception_v3): inception_v3 | resnet152',
                        choices=["inception_v3","resnet152"])
parser.add_argument('-d', default="ucf101", type=str,
                        help='dataset (default: ucf101): ucf101 | hmdb51',
                        choices=["hmdb51","ucf101"])
parser.add_argument('-m', default="rgb2", type=str,
                        help='modalities (default: rgb2)')
parser.add_argument('-o', default="output/", type=str,
                        help='directory to save the Histogram and confusion matrix (default: output/)')

def obtain_accuracies_per_class(data_complete, ground_truth, num_classes=101):
    class_count = np.zeros(num_classes)
    y_pred = np.zeros(len(data_complete))

    for i, data in enumerate(data_complete):
        index = np.argmax(data)
        y_pred[i] = index
        class_count[ground_truth[i]] += 1

    conf_matrix = confusion_matrix(ground_truth, y_pred)
    match_count = np.diagonal(conf_matrix)

    return match_count / class_count


def obtain_ground_truth(path_file, num_classes=101):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []

    for line in lines_file: 
        line_info = line.split(' ')
        ground_truth.append(int(line_info[2]))

    return np.array(ground_truth)


def plot_acc(npy_path, val_path, train_path, num_classes, split, output_path):#, modalities, parameters, split, use_fuzzy=False, softmax_norm=False):
    ground_truth = obtain_ground_truth(val_path%(split), num_classes)
    data = np.load(npy_path%(split))
    acc_np_plot = obtain_accuracies_per_class(data, ground_truth, num_classes)
    train_list = obtain_ground_truth(train_path%(split), num_classes)

    val_hist, _ = np.histogram(ground_truth, bins=num_classes)
    val_hist = val_hist/float(val_hist.max())
    train_hist, _ = np.histogram(train_list, bins=num_classes)
    train_hist = train_hist/float(train_hist.max())
    
    prev = 0
    for i in range(30,num_classes,30):
        n = len(train_hist[prev:i])
        labels = np.arange(n)
        plt.hlines(np.arange(0.2, 1.0, 0.2), xmin=-2, xmax=n+1, linestyles='dashed', color='black')
        plt.xlim(-2, n+1)
        plt.ylim(-0.02, 1.02)
        plt.xticks(np.arange(0, n, 5))
        plt.bar(labels, train_hist[prev:i], color='g', width=0.25)
        plt.bar(labels+0.25, val_hist[prev:i], color='r', width=0.25)
        plt.bar(labels+0.5, acc_np_plot[prev:i], color='b', width=0.25)
        plt.show()
        prev += 30




def main():
    args = parser.parse_args()

    # dataset, modality, architecture, split
    npy_path = os.path.join(args.npy_dir, "%s/%s_%s_%s_s%s.npy") % (args.d, args.d, args.m, args.a, "%s")
    val_path = os.path.join(args.splits_dir, "%s/val_split%s.txt" % (args.d, "%s"))
    train_path = os.path.join(args.splits_dir, "%s/train_split%s.txt" % (args.d, "%s"))
    num_classes = 101 if args.d == 'ucf101' else 51

    split = "_s%d" % args.split
    output_path = os.path.join(args.o, "%s_%s_%s%s.png" % (args.d, args.a, args.m, split))

    if not os.path.isdir(args.o):
        print("Creating output directory: ", args.o)
        os.makedirs(args.o)

    plot_acc(npy_path, val_path, train_path, num_classes, args.split, output_path)


if __name__ == '__main__':
    main()    
