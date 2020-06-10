##
## Diversity measures: http://repositorio.unicamp.br/jspui/bitstream/REPOSIP/275503/1/Faria_FabioAugusto_D.pdf
##      H1   M1
##  H2  a     b
##  M2  c     d
##
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
parser.add_argument('-m1', default="rgb2", type=str,
                        help='First modality (default: rgb2)')
parser.add_argument('-m2', default="flow", type=str,
                        help='Second modality (default: flow)')

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


def obtain_ground_truth(path_file):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []

    for line in lines_file: 
        line_info = line.split(' ')
        ground_truth.append(int(line_info[2]))

    np.array(ground_truth)
    return ground_truth


def div_measure(npy_path_1, npy_path_2, val_path, modality1, modality2):
    ground_truth = obtain_ground_truth(val_path)
    data1 = np.load(npy_path_1)
    data2 = np.load(npy_path_2)
    hm, acc1, acc2 = hit_miss(data1, data2, ground_truth)
    
    a,b,c,d = hm.flatten()
    
    # Diversity measures
    cor = (a*d - b*c) / (((a+b) * (c+d) * (a+c) * (b+d)) ** 0.5)
    dfm = d
    dm = (b+c)/(a+b+c+d)
    qstat = (a*d - b*c) / (a*d + b*c)
    ia = (2*(a*c - b*d)) / (((a+b) * (c+d)) + ((a+c) * (b+d)))

    # Complementarity
    comp12 = 1. - (d / (b+d))
    comp21 = 1. - (d / (c+d))

    report = "REPORT\n"
    report += "Accuracy 1 ({0}): {1:.4f}\n".format(modality1, acc1)
    report += "Accuracy 2 ({0}): {1:.4f}\n".format(modality2, acc2)
    report += "\n-------------------------------\n\n"
    report += "Hit and miss table:\n"
    report += "\t  H1\t  M1\n"
    report += "H2\t{:0.4f}\t{:0.4f}\n".format(hm[0][0], hm[0][1])
    report += "M2\t{:0.4f}\t{:0.4f}\n".format(hm[1][0], hm[1][1])
    report += "\n-------------------------------\n\n"
    report += "Diversity measures:\n"
    report += "(The lower, the better):\n"
    report += "COR(c1,c2) = {:0.4f}\n".format(cor)
    report += "DFM(c1,c2) = {:0.4f}\n".format(dfm)
    report += "QSTAT(c1,c2) = {:0.4f}\n".format(qstat)
    report += "IA(c1,c2) = {:0.4f}\n".format(ia)
    report += "\n(The greater, the better):\n"
    report += "DM(c1,c2) = {:0.4f}\n".format(dm)
    report += "\n-------------------------------\n\n"
    report += "Complementarity:\n"
    report += "(The greater, the better):\n"
    report += "Comp(c1,c2) = {:0.4f}\n".format(comp12)
    report += "Comp(c2,c1) = {:0.4f}\n".format(comp21)

    print(report)

def main():
    args = parser.parse_args()

    # dataset, modality, architecture, split
    npy_path_1 = os.path.join(args.npy_dir, "%s/%s_%s_%s_s%d.npy") % (args.d, args.d, args.m1, args.a, args.split)
    npy_path_2 = os.path.join(args.npy_dir, "%s/%s_%s_%s_s%d.npy") % (args.d, args.d, args.m2, args.a, args.split)
    val_path = os.path.join(args.val_dir, "%s/val_split%d.txt" % (args.d, args.split))
    num_classes = 101 if args.d == 'ucf101' else 51

    div_measure(npy_path_1, npy_path_2, val_path, args.m1, args.m2)


if __name__ == '__main__':
    main()    
