##
## Classifier combination: https://www.aclweb.org/anthology/P98-1029.pdf
## Complementarity Comp(A,B): frequency that B is correct when A is incorrect (Comp(A,B) != Comp(B,A))
##
##

import argparse
import itertools
import operator
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser(description='')
parser.add_argument('npy1', type=str, help='path to npy files')
parser.add_argument('npy2', type=str, help='path to npy files')
parser.add_argument('--settings', metavar='DIR', default='../datasets/settings_earlystop',
                    help='path to dataset setting files')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
          help='which split of data to work on (default: 1)')
parser.add_argument('-d', default="ucf101", type=str,
                        help='dataset (default: ucf101): ucf101 | hmdb51',
                        choices=["hmdb51","ucf101"])

def hit_miss(data1, data2, ground_truth):
    hit_miss = np.zeros((2,2))
    acc1, acc2 = 0, 0

    for i, (d1, d2) in enumerate(zip(data1, data2)):
        y_pred1 = np.argmax(d1)
        y_pred2 = np.argmax(d2)
        y = ground_truth[i]

        hit1 = y_pred1 == y
        hit2 = y_pred2 == y

        acc1 += 1 if hit1 else 0
        acc2 += 1 if hit2 else 0

        hit_miss[0][0] += int(hit1 and hit2)
        hit_miss[0][1] += int(not hit1 and hit2)
        hit_miss[1][0] += int(hit1 and not hit2)
        hit_miss[1][1] += int(not hit1 and not hit2)

    hit_miss /= len(data1)
    acc1 /= len(data1)
    acc2 /= len(data1)

    return hit_miss, acc1, acc2


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

def complementarity(a, b, c, d):
    comp12 = b / (b + d)
    comp21 = c / (c + d)
    harm_mean = (2 * comp12 * comp21) / (comp12 + comp21)

    return comp12, comp21, harm_mean

def kappa(a, b, c, d):
    m = a + b + c + d
    theta1 = (a + d) / m
    theta2 = (((a + b) * (a + c)) + ((d + b) * (d + c))) / (m**2)

    k = (theta1 - theta2) / (1. - theta2)

    return k


##         hit1   miss1
##  hit 2    a      b
##  miss 2   c      d

def measures(npy_path_1, npy_path_2, test_path):
    _, ts_labels = get_labels(test_path)
    data1 = np.load(npy_path_1)
    data2 = np.load(npy_path_2)
    hm, acc1, acc2 = hit_miss(data1, data2, ts_labels)
    
    a, b, c, d = hm.flatten().astype(float)
    
    comp12, comp21, harm_mean = complementarity(a, b, c, d)
    k = kappa(a, b, c, d)

    return comp12, comp21, harm_mean, k

def main():
    args = parser.parse_args()

    test_path = os.path.join(args.settings, "%s/test_split%s.txt" % (args.d, args.split))

    print(measures(args.npy1, args.npy2, test_path))


if __name__ == '__main__':
    main()    
