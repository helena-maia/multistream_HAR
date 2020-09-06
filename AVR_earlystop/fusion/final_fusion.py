from fusion import fixed_SVM, get_labels
import os
import numpy as np
import itertools

npys1 = np.loadtxt("npy_list.csv", dtype=str)
datasets = npys1[:, 5]

u_modalities = ["RGB", "FLOW", "AVR", "LVR0", "LVR1", "LVR2"]
u_datasets = ["ucf101", "hmdb51"]
u_splits = [1, 2, 3]
settings = "../../AVR_earlystop/datasets/settings_earlystop/"

combination = dict()
parameters = dict()

combination['ucf101'] = ["RGB","FLOW","AVR","LVR1","LVR2"]
parameters['ucf101'] = [{'C':10.**(-3), 'kernel': 'linear', 'gamma': 10**(-4), 'ms': 'ovr'},
                        {'C':10.**(-3), 'kernel': 'linear', 'gamma': 10**(-4), 'ms': 'ovr'},
                        {'C':10.**(-3), 'kernel': 'linear', 'gamma': 10**(-4), 'ms': 'ovr'}]
combination['hmdb51'] = ["RGB","FLOW","AVR","LVR0","LVR1","LVR2"]
parameters['hmdb51'] = [{'C':10., 'kernel': 'rbf', 'gamma': 10**(-3), 'ms': 'ovr'},
                        {'C':10.**(-1), 'kernel': 'linear', 'gamma': 10**(-4), 'ms': 'ovr'},
                        {'C':10.**(-2), 'kernel': 'linear', 'gamma': 10**(-4), 'ms': 'ovr'}]

for d in u_datasets:
    npys2 = npys1[datasets == d]
    splits = npys2[:, 3]

    for s in u_splits:
        print("svm_"+d+"_"+str(s)+".npy")

        npys3 = npys2[splits == str(s)]
        modalities = npys3[:, 4]

        npy_data = []
        for m in combination[d]:
            n = npys3[modalities == m][0]
            npy_path = os.path.join(n[0], n[1], n[2])
            npy_data.append(np.load(npy_path))

        split_path = os.path.join(settings, "%s/%s_split%s.txt" % (d, "%s", s))
        train_path = split_path%("train")
        val_path = split_path%("val")
        test_path = split_path%("test")
        dataset_path = os.path.join(settings, "%s/dataset_list.txt" % (d))

        ds_keys, ds_labels = get_labels(dataset_path)
        tr_keys, tr_labels = get_labels(train_path)
        vl_keys, vl_labels = get_labels(val_path)
        ts_keys, ts_labels = get_labels(test_path)

        tr_ind = [ds_keys.index(key) for ind, key in enumerate(tr_keys)]
        vl_ind = [ds_keys.index(key) for ind, key in enumerate(vl_keys)]
        ts_ind = [ds_keys.index(key) for ind, key in enumerate(ts_keys)]

        X_tr, X_vl, X_ts = [], [], []
        for npy in npy_data:
            tr, vl, ts = npy[tr_ind], npy[vl_ind], npy[ts_ind]
            X_tr.append(tr)
            X_vl.append(vl)
            X_ts.append(ts)

        X_tr, X_vl, X_ts = np.array(X_tr), np.array(X_vl), np.array(X_ts)
        y_tr, y_vl, y_ts = np.array(tr_labels), np.array(vl_labels), np.array(ts_labels)

        c, k = parameters[d][s-1]['C'], parameters[d][s-1]['kernel']
        g, m = parameters[d][s-1]['gamma'], parameters[d][s-1]['ms']

        y_pred, prec = fixed_SVM(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts, C=c, gamma=g, kernel=k, decision_function_shape=m)

        print(prec)
        np.save("svm_"+d+"_"+str(s)+".npy", y_pred)


