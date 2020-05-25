import argparse
import numpy as np

parser = argparse.ArgumentParser(description='')
parser.add_argument('cm_path', type=str, help='path to npy files')
parser.add_argument('-k', default=10, type=int, help='Top k confusions')
parser.add_argument('-class_index_path', default="splits/ucf101/classInd.txt", type=str, 
	                help='path to class_index file (default: splits/ucf101/classInd.txt)')

def top_k_conf(cm, k = 10):
    pair_dict = {}

    for i, row in enumerate(cm):
        for j, cell in enumerate(row):
            if i != j:
                pair_dict[(i,j)] = cell

    pair_sorted = sorted(pair_dict, key = pair_dict.get)
    topk = {key: pair_dict[key] for key in pair_sorted[-k:][::-1]}

    return topk
    

def main():
    args = parser.parse_args()
    conf_matrix = np.load(args.cm_path)
    topk = top_k_conf(conf_matrix, k = args.k)

    class_ind = np.loadtxt(args.class_index_path, dtype = "U200")
    class_dict = {int(l[0]): l[1] for l in class_ind}

    print("Top", args.k)
    for pair in topk:
    	print(class_dict[pair[0]], class_dict[pair[1]], topk[pair])

    




if __name__ == '__main__':
    main()
