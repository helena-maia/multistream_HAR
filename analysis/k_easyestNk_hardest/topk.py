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
    accum_acc = np.zeros(num_classes, dtype=float)

    for s in range(1,4):
        test_path = os.path.join(args.settings, "%s/test_split%s.txt" % (args.d, s))
        _, ts_labels = get_labels(test_path)
        npy_data = np.load(args.npy_paths[s-1])
        acc_class, _ = obtain_accuracies_per_class(npy_data, ts_labels, num_classes)

        accum_acc += acc_class

    accum_acc /= 3
    indices = np.argsort(accum_acc)

    class_path = os.path.join(args.settings, "%s/class_ind.txt" % (args.d))
    class_ind = np.loadtxt(class_path, dtype=str)
    k_hardest = indices[:args.k]
    k_easiest = indices[-args.k:]
    
    k_hardest_label = class_ind[k_hardest][:,1]
    k_hardest_acc = accum_acc[k_hardest]

    k_easiest_label = class_ind[k_easiest][:,1]
    k_easiest_acc = accum_acc[k_easiest]


    plt.yticks(np.arange(2*args.k), np.concatenate((k_hardest_label, k_easiest_label)), fontsize=22)
    plt.xticks(fontsize=22)
    p1 = plt.barh(np.arange(args.k), k_hardest_acc*100., color='red')
    p2 = plt.barh(np.arange(args.k, 2*args.k), k_easiest_acc*100., color='blue')
    plt.legend([p1,p2], ["Hardest", "Easiest"], loc="lower right", title=None, ncol=1, fontsize=22)
    #plt.show()
    
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    fig.savefig(args.o, dpi=100, bbox_inches='tight')
