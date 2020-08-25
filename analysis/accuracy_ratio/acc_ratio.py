import numpy as np
from class_acc import class_acc

npy_list = np.loadtxt("npy_list.csv", dtype=str)
datasets = npy_list[:, 5]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"

class_acc_list = [[],[]]

for i, d in enumerate(u_datasets):
    npy_d = npy_list[datasets == d]
    modalities = npy_d[:,4]

    for j, m in enumerate(u_modalities):
        npy_m = npy_d[modalities == m]
        npy_paths = []
        npy_paths.append(os.path.join(npy_m[0][0], npy_m[0][1], npy_m[0][2]))
        npy_paths.append(os.path.join(npy_m[1][0], npy_m[1][1], npy_m[1][2]))
        npy_paths.append(os.path.join(npy_m[2][0], npy_m[2][1], npy_m[2][2]))
        class_acc_list[i].append(class_acc(d, settings, npy_paths))

class_acc_list = np.array(class_acc_list)
print(class_acc_list.shape)




