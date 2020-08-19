import numpy as np
import os
from complementarity import complementarity

npy_list = np.loadtxt("npy_list.csv", dtype=str)
splits = npy_list[:, 3]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"

accum_compl = [np.zeros((6,6), dtype=float), np.zeros((6,6), dtype=float)]

for s in range(1, 4):
	npys = npy_list[splits == str(s)]
	datasets = npys[:, 5]

	for i, d in enumerate(u_datasets):
		compl = np.zeros((6,6))
		npys2 = npys[datasets == d]
		modalities = npys2[:, 4]

		test_path = os.path.join(settings, "%s/test_split%d.txt" % (d, s))

		for j, m in enumerate(u_modalities):
			for k in range(j+1, 6):
				row1 = npys2[modalities == m][0]
				row2 = npys2[modalities == u_modalities[k]][0]

				npy_path_1 = os.path.join(row1[0], row1[1], row1[2])
				npy_path_2 = os.path.join(row2[0], row2[1], row2[2])

				comp12, comp21, harm_mean = complementarity(npy_path_1, npy_path_2, test_path)
				accum_compl[i][j,k] += comp12
				accum_compl[i][k,j] += comp21

accum_compl[0] /= 3
accum_compl[1] /= 3
print(accum_compl)