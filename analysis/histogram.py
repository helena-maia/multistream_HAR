import argparse
import itertools
import operator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='')
parser.add_argument('npy_dir', type=str, help='path to npy files')
parser.add_argument('val_dir', type=str, help='path to npy files')
parser.add_argument('-s', '--split', default=-1, type=int, metavar='S',
          help='which split of data to work on (default: -1)')
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

    return match_count / class_count, conf_matrix


def obtain_ground_truth(path_file, num_classes=101):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []

    for line in lines_file: 
        line_info = line.split(' ')
        ground_truth.append(int(line_info[2]))

    np.array(ground_truth)
    return ground_truth


def plot_acc(npy_path, val_path, num_classes, split, output_path, cm_path):#, modalities, parameters, split, use_fuzzy=False, softmax_norm=False):
    if split == -1:
        acc_np_plot = np.zeros(num_classes)

        for s in range(3):
            ground_truth = obtain_ground_truth(val_path%(s+1), num_classes)
            data = np.load(npy_path%(s+1))
            acc_np, conf_matrix = obtain_accuracies_per_class(data, ground_truth, num_classes)
            np.save(cm_path % (s+1), conf_matrix)
            print("Confusion matrix saved: ", cm_path%s)
            acc_np_plot += acc_np

        acc_np_plot /= 3
    else:
        ground_truth = obtain_ground_truth(val_path%(split), num_classes)
        data = np.load(npy_path%(split))
        acc_np_plot, conf_matrix = obtain_accuracies_per_class(data, ground_truth, num_classes)
        np.save(cm_path%split, conf_matrix)
        print("Confusion matrix saved: ", cm_path%split)

    print(np.where(acc_np_plot < 0.2))
    
    plt.hlines(np.arange(20, 100, 20), xmin=-2, xmax=num_classes+1, linestyles='dashed', color='black')
    plt.xlim(-2,num_classes+1)
    plt.ylim(-2,102)
    plt.xticks(np.arange(0, num_classes, 5))
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.bar(np.arange(num_classes), acc_np_plot*100.)

    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    fig.savefig(output_path, dpi=100)

    print("Histogram saved:", output_path)



def main():
    args = parser.parse_args()

    # dataset, modality, architecture, split
    npy_path = os.path.join(args.npy_dir, "%s/%s_%s_%s_s%s.npy") % (args.d, args.d, args.m, args.a, "%s")
    val_path = os.path.join(args.val_dir, "%s/val_split%s.txt" % (args.d, "%s"))
    num_classes = 101 if args.d == 'ucf101' else 51

    split = "_s%d" % args.split if args.split != -1 else ""
    output_path = os.path.join(args.o, "%s_%s_%s%s.eps" % (args.d, args.a, args.m, split))
    cm_path = os.path.join(args.o, "conf_matrix_%s_%s_%s_s%s.npy" % (args.d, args.a, args.m, "%d"))

    if not os.path.isdir(args.o):
        print("Creating output directory: ", args.o)
        os.makedirs(args.o)

    plot_acc(npy_path, val_path, num_classes, args.split, output_path, cm_path)


if __name__ == '__main__':
    main()    
