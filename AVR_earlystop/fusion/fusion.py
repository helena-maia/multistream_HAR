import argparse
import os
import numpy as np
from sklearn.metrics import precision_score
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from fuzzy_fusion import fuzzyFusion_2 as fuzzy_fusion_choquet, fuzzy_fusion_sugeno
from fc_fusion import fc_fusion
from sklearn.preprocessing import StandardScaler

def get_labels(path_file):
    file_ = open(path_file, "r")
    file_lines = file_.readlines()

    name_list = []
    label_list = []

    for line in file_lines:
        line_info = line.split(' ')
        video_name = line_info[0]
        label = int(line_info[2])

        name_list.append(video_name)
        label_list.append(label)

    return name_list, label_list

methods = ["individual", "simple_avg", "fixed_weighted_avg", "weighted_avg", "choquet_fuzzy", "sugeno_fuzzy", "FC", "SVM"]

def individual(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    prec_list = []
    n_modalities = len(X_ts)

    for i in range(n_modalities):
        y_pred = np.argmax(X_ts[i], axis=1)

        # Multiclass precision: calculate metrics globally by counting the total true positives
        prec = precision_score(y_ts, y_pred, average ='micro')
        prec_list.append(prec)

    return (None, None, prec_list)

def simple_avg(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    n_samples = y_ts.shape[0]
    X_comb = np.mean(X_ts, axis=0) # n_samples X n_classes

    y_pred = np.argmax(X_comb, axis=1) # n_samples

    # Multiclass precision: calculate metrics globally by counting the total true positives
    prec = precision_score(y_ts, y_pred, average ='micro')

    return (None, None, prec)

def fixed_weighted_avg(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    n_samples = y_ts.shape[0]
    w = [1.,2.]
    X_comb = fuzzy_fusion(X_ts, w)
    #X_w = [X_ts[i] * w[i] for i in range(2)]
    #X_comb = np.mean(X_w, axis=0) # n_samples X n_classes


    y_pred = np.argmax(X_comb, axis=1) # n_samples

    # Multiclass precision: calculate metrics globally by counting the total true positives
    prec = precision_score(y_ts, y_pred, average ='micro')

    return (None, None, prec)


def iter_weights(n_modalities):
    # Linear and exponential weights
    '''
    linear = [i+1 for i in range(n_modalities)]
    linear_weights = itertools.permutations(linear, n_modalities)
    exp = [2**i for i in range(n_modalities)]
    exp_weights = itertools.permutations(exp, n_modalities)

    weights = itertools.chain(linear_weights, exp_weights)
    '''

    # GridSearch - v1
    #weights = itertools.product(range(5,100,5),repeat=n_modalities)

    # GridSearch - v2
    weights = itertools.product(range(1,10,1),repeat=n_modalities)

    return weights

def weighted_avg(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.array([np.concatenate((X1,X2),axis=0) for X1, X2 in zip(X_tr, X_vl)])
    y_tr_ = np.concatenate((y_tr, y_vl), axis=0)
    n_modalities = len(X_tr)

    def weighted_avg_step(X, y, w):
        X_w = [X[i] * w[i] for i in range(n_modalities)]
        X_comb = np.mean(X_w, axis=0) # n_samples X n_classes
        y_pred = np.argmax(X_comb, axis=1) # n_samples
        prec = precision_score(y, y_pred, average ='micro')

        return prec

    max_prec = 0
    best_weight = None

    weights = iter_weights(n_modalities)

    for w in weights:
        prec = weighted_avg_step(X_tr_, y_tr_, w)
        if prec > max_prec:
            max_prec = prec
            best_weight = w

    prec = weighted_avg_step(X_ts, y_ts, best_weight)

    return (best_weight, max_prec, prec)

def choquet_fuzzy(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.array([np.concatenate((X1,X2),axis=0) for X1, X2 in zip(X_tr, X_vl)])
    y_tr_ = np.concatenate((y_tr, y_vl), axis=0)
    n_modalities = len(X_tr)

    def choquet_fuzzy_step(X, y, w):
        X_comb = fuzzy_fusion_choquet(X, w)
        y_pred = np.argmax(X_comb, axis=1) # n_samples
        prec = precision_score(y, y_pred, average ='micro')

        return prec

    weights = iter_weights(n_modalities)

    max_prec = 0
    best_weight = None

    for w in weights:
        w = [x/10. for x in w]
        prec = choquet_fuzzy_step(X_tr_, y_tr_, w)

        if prec > max_prec:
            max_prec = prec
            best_weight = w

    prec = choquet_fuzzy_step(X_ts, y_ts, best_weight)

    return (best_weight, max_prec, prec)

def sugeno_fuzzy(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.array([np.concatenate((X1,X2),axis=0) for X1, X2 in zip(X_tr, X_vl)])
    y_tr_ = np.concatenate((y_tr, y_vl), axis=0)
    n_modalities = len(X_tr)

    def sugeno_fuzzy_step(X, y, w):
        w = [x/10. for x in w]
        X_comb = fuzzy_fusion_sugeno(X, w)
        y_pred = np.argmax(X_comb, axis=1) # n_samples
        prec = precision_score(y, y_pred, average ='micro')

        return prec

    weights = iter_weights(n_modalities)

    max_prec = 0
    best_weight = None

    for w in weights:
        prec = sugeno_fuzzy_step(X_tr_, y_tr_, w)
        continue

        if prec > max_prec:
            max_prec = prec
            best_weight = w

    return None
    prec = sugeno_fuzzy_step(X_ts, y_ts, best_weight)

    return (best_weight, max_prec, prec)

def FC(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.concatenate((X_tr), axis=1)
    X_vl_ = np.concatenate((X_vl), axis=1)
    X_ts_ = np.concatenate((X_ts), axis=1)

    n_modalities, n_samples, n_classes = X_tr.shape
    #arq = [('L', n_modalities*n_classes, n_classes), ('R'), ('D',0.9)]
    arq = [('L', n_modalities*n_classes, n_classes), ('R')]

    prec = fc_fusion(X_tr_, X_vl_, X_ts_, y_tr, y_vl, y_ts, arq = arq)/100.

    return (None, None, prec)

def SVM(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.array([np.concatenate((X1,X2),axis=0) for X1, X2 in zip(X_tr, X_vl)])
    X_tr_ = np.concatenate((X_tr_), axis=1)
    y_tr_ = np.concatenate((y_tr, y_vl), axis=0)

    X_ts_ = np.concatenate((X_ts), axis=1)

    clf = SVC(random_state=42)
    parameters = {
        'C': list(10.**np.arange(-5,6)),
        'gamma': list(10.**np.arange(-10,11)),
        'kernel': ['rbf', 'linear'],
        'decision_function_shape': ['ovr', 'ovo']
    }

    scaler = StandardScaler()
    X_tr_ = scaler.fit_transform(X_tr_)
    X_ts_ = scaler.transform(X_ts_)

    gs = GridSearchCV(clf, parameters, n_jobs=30, verbose=1, scoring='accuracy', cv=3)
    gs.fit(X_tr_, y_tr_)

    best_clf = gs.best_estimator_
    best_param = gs.best_params_
    best_score = gs.best_score_

    y_pred = best_clf.predict(X_ts_)
    prec = precision_score(y_ts, y_pred, average ='micro')

    return (best_param, best_score, prec)


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Three-Stream Action Recognition - Fusion')
    parser.add_argument('npy_paths', type=str, nargs='+',
                        help='path to npy files (list)')
    parser.add_argument('-s', default=1, type=int, metavar='S',
                        help='which split of data to work on (default: 1): 1 | 2 | 3',
                        choices=[1,2,3])
    parser.add_argument('-d', default="ucf101", type=str, metavar='DATASET',
                        help='dataset (default: ucf101): ucf101 | hmdb51',
                        choices=["hmdb51","ucf101"])
    parser.add_argument('-m', metavar='METHOD', default='simple_avg', 
                        help='fusion method:'+' | '.join(methods) +' (default: '+ methods[0] +') ',
                        choices=methods)
    parser.add_argument('--settings', metavar='DIR', default='../datasets/settings_earlystop',
                        help='path to dataset setting files')

    return parser.parse_args()

def fusion(args):
    split_path = os.path.join(args.settings, "%s/%s_split%s.txt" % (args.d, "%s", args.s))
    train_path = split_path%("train")
    val_path = split_path%("val")
    test_path = split_path%("test")
    dataset_path = os.path.join(args.settings, "%s/dataset_list.txt" % (args.d))

    npy_data = []
    for npy_path in args.npy_paths:
        npy_data.append(np.load(npy_path))

    if len(npy_data) < 2: return None

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

    fusion_call = args.m + "(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts)"
    ret = eval(fusion_call)

    return ret


if __name__ == '__main__':
    args = get_args()
    ret = fusion(args)

    if ret:
        best_param, best_score, prec = ret

        if best_param: print("Best parameters:", best_param)
        if best_score: print("Best precision:", best_score)
        if isinstance(prec, list):
            print("Prec:")
            for n,p in zip(args.npy_paths, prec):
                print("\t{}: {:.04f}".format(n,p))
        elif isinstance(prec, float): print("Prec: {:.04f}".format(prec))
    else: print("Missing npy file")


