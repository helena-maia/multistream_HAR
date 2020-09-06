import numpy as np
import os
from class_acc import class_acc, class_acc2
import matplotlib.pyplot as plt

npy_list = np.loadtxt("npy_list.csv", dtype=str)
datasets = npy_list[:, 5]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"
colors = plt.get_cmap('Dark2').colors
markers = ['s','o','d','v','x','+'] 

for i, d in enumerate(u_datasets):
    class_path = os.path.join(settings, "%s/class_ind.txt" % (d))
    class_ind = np.loadtxt(class_path, dtype=str)

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

    fusion_paths = ["svm_"+d+"_1.npy", "svm_"+d+"_2.npy", "svm_"+d+"_3.npy"]
    fusion_acc = class_acc2(d, settings, fusion_paths)

    
    max_acc = np.max(class_acc_list, axis=0)
    asort = (np.argsort(max_acc)[::-1])[:51]
    fusion_acc2 = fusion_acc[asort]*100
    max_acc2 = max_acc[asort]*100

    bottom = []
    diff = []
    color_diff = []
    for x in zip(max_acc2, fusion_acc2):
        b, t, c = (x[0], x[1], "green") if x[0] < x[1] else (x[1], x[0], "red")
        bottom.append(b)
        diff.append(t-b)
        color_diff.append(c)

    class_acc_list = np.array([class_acc_list[:,ind] for ind in asort]).transpose()
    class_ind = np.array([class_ind[ind] for ind in asort])[:,1]

    plt.ylim(bottom=0., top=101.)
    plt.xlim(left=-1, right=len(max_acc2))
    plt.vlines(np.arange(0,len(max_acc2)), ymin=0, ymax=100, linestyles='dashed', color='lightgray', zorder=-1)
    plt.ylabel('Recall', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.xticks(np.arange(0,len(max_acc2)), class_ind, rotation='vertical')
    plt.yticks(np.arange(0,101,20))
    plt.bar(np.arange(len(max_acc2)), diff, bottom=bottom, color=color_diff)


    fig = plt.gcf()
    fig.set_size_inches(20,9)
    fig.savefig("plot{}.eps".format(i), dpi=100, bbox_inches='tight')
    fig.clear()
