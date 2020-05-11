import argparse
import itertools
import operator
import numpy as np
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('npy_dir', type=str, help='path to npy files')
parser.add_argument('val_dir', type=str, help='path to npy files')
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


def obtain_accuracies_per_class(data_complete, ground_truth, num_classes=101):
    class_count = np.zeros(num_classes)
    match_count = np.zeros(num_classes)

    for i, data in enumerate(data_complete):
        index = np.argmax(data)
        class_count[ground_truth[i]] += 1

        if ground_truth[i] == index:
            match_count[ground_truth[i]] += 1
    
    return match_count / class_count


def obtain_ground_truth(path_file, num_classes=101):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []

    for line in lines_file: 
        line_info = line.split(' ')
        ground_truth.append(int(line_info[2]))

    np.array(ground_truth)
    return ground_truth


def plot_acc(npy_path, val_path, num_classes):#, modalities, parameters, split, use_fuzzy=False, softmax_norm=False):
    ground_truth = obtain_ground_truth(val_path, num_classes)

    data = np.load(npy_path)
    acc_np = obtain_accuracies_per_class(data, ground_truth, num_classes)

    print(acc_np[30])

    plt.xticks(np.arange(0, num_classes, 5))
    plt.bar(np.arange(num_classes), acc_np)
    plt.show()


def main():
    args = parser.parse_args()

    # dataset, modality, architecture, split
    npy_path = os.path.join(args.npy_dir, "%s_%s_%s_s%s.npy") % (args.d, args.m, args.a, args.split)
    val_path = os.path.join(args.val_dir, "%s/val_split%s.txt" % (args.d, args.split))
    num_classes = 101 if args.d == 'ucf101' else 51
	
    plot_acc(npy_path, val_path, num_classes) #, args.m, w, s, args.f, args.n)


if __name__ == '__main__':
    main()    
