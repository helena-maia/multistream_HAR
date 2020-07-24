import argparse
import os
import numpy as np
from sklearn.metrics import precision_score

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


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Three-Stream Action Recognition - NPY Test')
    parser.add_argument('npy_path', type=str,
                        help='path to npy files (list)')
    parser.add_argument('-s', default=1, type=int, metavar='S',
                        help='which split of data to work on (default: 1): 1 | 2 | 3',
                        choices=[1,2,3])
    parser.add_argument('-d', default="ucf101", type=str, metavar='DATASET',
                        help='dataset (default: ucf101): ucf101 | hmdb51',
                        choices=["hmdb51","ucf101"])
    parser.add_argument('--settings', metavar='DIR', default='../datasets/settings_earlystop',
                        help='path to dataset setting files')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    test_path = os.path.join(args.settings, "%s/test_split%s.txt" % (args.d, args.s))
    _, ts_labels = get_labels(test_path)
    npy_data = np.load(args.npy_path)
    y_pred = np.argmax(npy_data, axis=1)

    # Multiclass precision: calculate metrics globally by counting the total true positives
    prec = precision_score(ts_labels, y_pred, average ='micro')

    print("Prec: {:.04f}".format(prec))


