import argparse
import os
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def get_labels(path_file):
    file_ = open(path_file, "r")
    file_lines = file_.readlines()

    name_list = []
    label_list = []

    for line in file_lines:
        line_info = line.split(' ')
        video_name = line_info[0]
        label = int(line_info[2])

        name_list.append(video_name)
        label_list.append(label)

    return name_list, label_list

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


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('npy_paths', type=str, nargs='+',
                        help='path to npy files (list)')
    parser.add_argument('-k', default=10, type=int,
                        help='top k')
    parser.add_argument('-d', default="ucf101", type=str, metavar='DATASET',
                        help='dataset (default: ucf101): ucf101 | hmdb51',
                        choices=["hmdb51","ucf101"])
    parser.add_argument('--settings', metavar='DIR', default='../datasets/settings_earlystop',
                        help='path to dataset setting files')
    parser.add_argument('-o', default="img.eps", type=str,
                        help='output')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    num_classes = 101 if args.d == "ucf101" else 51
    accum_conf = np.zeros((num_classes,num_classes), dtype=float)

    for s in range(1,4):
        test_path = os.path.join(args.settings, "%s/test_split%s.txt" % (args.d, s))
        _, ts_labels = get_labels(test_path)
        npy_data = np.load(args.npy_paths[s-1])
        _, conf_matrix = obtain_accuracies_per_class(npy_data, ts_labels, num_classes)

        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] #row normalization

        accum_conf += conf_matrix

    accum_conf /= 3
    pair_dict = dict()

    for i, row in enumerate(accum_conf):
        for j, cell in enumerate(row):
            if i != j:
                pair_dict[(i,j)] = cell

    pair_sorted = sorted(pair_dict, key = pair_dict.get)

    class_path = os.path.join(args.settings, "%s/class_ind.txt" % (args.d))
    class_ind = np.loadtxt(class_path, dtype=str)


    for i, k in enumerate(pair_sorted[-args.k:][::-1]): 
    	class1 = class_ind[k[0]][1]
    	class2 = class_ind[k[1]][1]
    	print("{} & {} & {:.2f} \\\\".format(class1, class2, pair_dict[k]*100))
    