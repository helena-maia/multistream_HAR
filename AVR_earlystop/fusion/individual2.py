from fusion import fusion, methods
import argparse
import os
import numpy as np

npys1 = np.loadtxt("npy_list.csv", dtype=str)
datasets = npys1[:, 5]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
u_splits = [1, 2, 3]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"

print("here")

for d in u_datasets:
	npys2 = npys1[datasets == d]
	splits = npys2[:, 3]

	for s in u_splits:
		npys3 = npys2[splits == str(s)]

		npy_paths = []
		modalities = []
		for n in npys3:
			npy_paths.append(os.path.join(n[0], n[1], n[2]))
			modalities.append(n[4])

		args = argparse.Namespace(d=d, m="individual", npy_paths=npy_paths,
                                  s=s, settings=settings)

		_, _, prec = fusion(args)
		print(d, s, prec)
