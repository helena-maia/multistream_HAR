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
parser.add_argument('-m', default=["rgb2","flow"], type=str, nargs='+',
                        help='Modalities (default: rgb2, flow)')

def complementarity(data, ground_truth, combinations):
    n_mod = len(data)
    n_videos = len(data[0])
    hit = np.zeros(n_mod)
    common = np.zeros((len(combinations), 3))

    for i in range(n_videos):
        y = ground_truth[i]

        for m in range(n_mod):
            d = data[m][i]
            y_pred = np.argmax(d)
            hit[m] = y_pred == y

        for j, (A, B) in enumerate(combinations):
            commonA = True
            for ind in A:
                if hit[ind]: 
                    commonA = False
                    break

            commonB = True
            for ind in B:
                if hit[ind]:
                    commonB = False
                    break

            common[j][0] += int(commonA)
            common[j][1] += int(commonB)
            common[j][2] += int(commonA and commonB)

    comp = np.zeros((len(combinations), 2))
    for i, com in enumerate(common):
        comp[i][0] = 1 - (common[i][2] /common[i][0]) # Comp(A,B)
        comp[i][1] = 1 - (common[i][2] /common[i][1]) # Comp(B,A)

    return comp


def obtain_ground_truth(path_file):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []

    for line in lines_file: 
        line_info = line.split(' ')
        ground_truth.append(int(line_info[2]))

    np.array(ground_truth)
    return ground_truth


def div_measure(npy_paths, val_path, modalities):
    ground_truth = obtain_ground_truth(val_path)
    data = []
    for npy_path in npy_paths:
        scores = np.load(npy_path)
        data.append(scores)
    
    n = len(npy_paths)
    combinations = []
    indices = list(range(0,n))

    all_comb = itertools.combinations(indices, 2)
    for c in all_comb:
        comb = list(c)
        combinations.append([[comb[0]],[comb[1]]])

    for r in range(2, n):
        for i in range(n):
            combined = indices[:i]+indices[i+1:]
            all_comb = itertools.combinations(combined, r)

            for c in all_comb:
                combinations.append([[i], list(c)])

    comp = complementarity(data, ground_truth, combinations)

    comp_combined = np.zeros(len(combinations))
    for i, (A,B) in enumerate(combinations):
        comp_combined[i] = min(comp[i][0],comp[i][1])

    indices = np.argsort(comp_combined)
    topk = indices[-10:]
    for ind in topk:
        print(combinations[ind], comp[ind][0], comp[ind][1], comp_combined[ind])



def main():
    args = parser.parse_args()

    if len(args.m) >= 2:
        npy_paths = []
        npy_path_fmt = os.path.join(args.npy_dir, "%s/%s_%s_%s_s%d.npy") % (args.d, args.d, "%s", args.a, args.split)
        for modality in args.m:
            npy_paths.append(npy_path_fmt%modality)

        val_path = os.path.join(args.val_dir, "%s/val_split%d.txt" % (args.d, args.split))
        num_classes = 101 if args.d == 'ucf101' else 51

        div_measure(npy_paths, val_path, args.m)


    


if __name__ == '__main__':
    main()    
