from fusion import fusion, methods
import argparse
import os
import numpy as np
import itertools

npys1 = np.loadtxt("npy_list.csv", dtype=str)
datasets = npys1[:, 5]

u_modalities = ["RGB", "FLOW", "AVR"]#, "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
u_splits = [1, 2, 3]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"
method = "weighted_avg"

n_modalities = len(u_modalities)
comb = None
for r in range(2, n_modalities+1): # combinations of 2 or more modalities
    c = itertools.combinations(u_modalities, r=r)
    comb = itertools.chain(comb, c) if comb else c

for c in comb:		
	for d in u_datasets:
		print(d)
		npys2 = npys1[datasets == d]
		splits = npys2[:, 3]

		for s in u_splits:
			print(s)
			npys3 = npys2[splits == str(s)]
			modalities = npys3[:, 4]

			npy_paths = []
			for m in c:
				n = npys3[modalities == m][0]
				npy_paths.append(os.path.join(n[0], n[1], n[2]))

			args = argparse.Namespace(d=d, m=method, npy_paths=npy_paths, s=s, settings=settings)

			_, _, prec = fusion(args)
			print(prec)
