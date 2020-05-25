import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('npy', type=str, help='path to npy files')

def obtain_ground_truth(path_file, num_classes=101):
    file_ = open(path_file, "r")
    lines_file = file_.readlines()

    ground_truth = []

    for line in lines_file: 
        line_info = line.split(' ')
        ground_truth.append(int(line_info[2]))

    np.array(ground_truth)
    return ground_truth

def main():
    args = parser.parse_args()
    X = np.load(args.npy)

    print("Running TSNE")
    X_tsne = TSNE(n_components = 2).fit_transform(X)
    print("Done")

    ground_truth = obtain_ground_truth("splits/hmdb51/val_split1.txt", 51)

    plt.scatter(X_tsne[:,0], X_tsne[:,1], c = ground_truth)
    plt.title('')
    plt.xlabel('')
    plt.xlabel('')
    plt.show()

if __name__ == '__main__':
    main()
