import numpy as np
import os
from get_example import get_example
import matplotlib.pyplot as plt

npy_list = np.loadtxt("npy_list.csv", dtype=str)
datasets = npy_list[:, 5]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"
class_indices_pos = dict()
class_indices_neg = dict()
class_indices_pos["ucf101"] = [62,11,11,96,11,11]
class_indices_neg["ucf101"] = [55,57,66,55,55,37]
class_indices_pos["hmdb51"] = [15,15,00,15,15,35]
class_indices_neg["hmdb51"] = [43,47,43,50,43,43]

for i, d in enumerate(u_datasets):  
    npy_d = npy_list[datasets == d]
    modalities = npy_d[:,4]

    for j, m in enumerate(u_modalities):
        npy_m = npy_d[modalities == m]
        npy_path = os.path.join(npy_m[0][0], npy_m[0][1], npy_m[0][2]) # Only first split
        
        pos = get_example(d, settings, npy_path, class_indices_pos[d][j], correct = True)
        neg = get_example(d, settings, npy_path, class_indices_neg[d][j], correct = False)
        
        np.savetxt("settings_gradCAM/%s/test_split1_%s.txt"%(d,m), [pos,neg], fmt="%s")