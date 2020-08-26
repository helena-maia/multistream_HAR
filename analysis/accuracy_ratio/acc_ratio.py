import numpy as np
import os
from class_acc import class_acc
import matplotlib.pyplot as plt

npy_list = np.loadtxt("npy_list.csv", dtype=str)
datasets = npy_list[:, 5]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"

for i, d in enumerate(u_datasets):
    class_acc_list = []
    npy_d = npy_list[datasets == d]
    modalities = npy_d[:,4]

    for j, m in enumerate(u_modalities):
        npy_m = npy_d[modalities == m]
        npy_paths = []
        npy_paths.append(os.path.join(npy_m[0][0], npy_m[0][1], npy_m[0][2]))
        npy_paths.append(os.path.join(npy_m[1][0], npy_m[1][1], npy_m[1][2]))
        npy_paths.append(os.path.join(npy_m[2][0], npy_m[2][1], npy_m[2][2]))
        class_acc_list.append(class_acc(d, settings, npy_paths))

    class_acc_list = np.array(class_acc_list)

    p = []
    for k in range(6):
        p = plt.scatter(np.arange(0,class_acc_list.shape[1]), class_acc_list[k])

    plt.legend(p, u_modalities)
    plt.savefig("plot{}.eps".format(i))

    '''
    print(class_acc_list[:,0])
    sort_mod = np.argsort(class_acc_list, axis=0)
    mean = np.mean(class_acc_list, axis=0)
    dev = np.std(class_acc_list, axis=0)
    class_acc_list = -np.power(class_acc_list - mean, 2.)
    class_acc_list = class_acc_list / (2. * np.power(dev, 2.))
    class_acc_list = 1. - np.exp(class_acc_list)


    for i in range(sort_mod.shape[1]):
        max, min = sort_mod[5][i], sort_mod[0][i]
        print(i, max, class_acc_list[max][i], min, class_acc_list[min][i]) 
        if max in [2,3,4,5]: 
        print("{}: {} {:.2f} - {} {:.2f}".format(i, u_modalities[max], class_acc_list[max][i], u_modalities[min], class_acc_list[min][i]))
    '''

    '''
    for i in range(sort_mod.shape[1]):
        max1, max2 = sort_mod[5][i], sort_mod[4][i]
        min1, min2 = sort_mod[0][i], sort_mod[1][i]
        max_ratio = 1. - class_acc_list[max2][i] / class_acc_list[max1][i]
        min_ratio = class_acc_list[min1][i] / class_acc_list[min2][i]

        if max_ratio < 0.2: print(max1, max2, i)
        if min_ratio < 0.2: print(min1, min2, i)
    '''






