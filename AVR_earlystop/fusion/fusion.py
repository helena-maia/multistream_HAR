import argparse
import os
import numpy as np
from sklearn.metrics import precision_score
import itertools
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from fuzzy_fusion import fuzzy_fusion
from fc_fusion import fc_fusion
#import operator
#from fuzzy_fusion import fuzzy_fusion


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

methods = ["simple_avg", "weighted_avg", "FC", "SVM", "choquet_fuzzy", "sugeno_fuzzy"]

def simple_avg(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    n_samples = y_ts.shape[0]
    X_comb = np.mean(X_ts, axis=0) # n_samples X n_classes

    y_pred = np.argmax(X_comb, axis=1) # n_samples

    # Multiclass precision: calculate metrics globally by counting the total true positives
    prec = precision_score(y_ts, y_pred, average ='micro')

    return prec

def iter_weights(n_modalities):
    linear = [i+1 for i in range(n_modalities)]
    linear_weights = itertools.permutations(linear, n_modalities)
    exp = [2**i for i in range(n_modalities)]
    exp_weights = itertools.permutations(exp, n_modalities)

    weights = itertools.chain(linear_weights, exp_weights)

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
            print("Update:", w, prec)
            max_prec = prec
            best_weight = w

    prec = weighted_avg_step(X_ts, y_ts, best_weight)

    return prec

def choquet_fuzzy(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.array([np.concatenate((X1,X2),axis=0) for X1, X2 in zip(X_tr, X_vl)])
    y_tr_ = np.concatenate((y_tr, y_vl), axis=0)
    n_modalities = len(X_tr)

    def choquet_fuzzy_step(X, y, w):
        X_comb = fuzzy_fusion(X, w)
        y_pred = np.argmax(X_comb, axis=1) # n_samples
        prec = precision_score(y, y_pred, average ='micro')

        return prec

    weights = iter_weights(n_modalities)

    max_prec = 0
    best_weight = None

    for w in weights:
        prec = choquet_fuzzy_step(X_tr_, y_tr_, w)

        if prec > max_prec:
            print("Update:", w, prec)
            max_prec = prec
            best_weight = w

    prec = choquet_fuzzy_step(X_ts, y_ts, best_weight)

    return prec

def sugeno_fuzzy(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    return

def FC(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.concatenate((X_tr), axis=1)
    X_vl_ = np.concatenate((X_vl), axis=1)
    X_ts_ = np.concatenate((X_ts), axis=1)

    n_modalities, n_samples, n_classes = X_tr.shape
    arq = [('L', n_modalities*n_classes, n_classes), ('R'), ('D',0.9)]

    prec = fc_fusion(X_tr_, X_vl_, X_ts_, y_tr, y_vl, y_ts, arq = arq)
    #return prec
    return 0

def SVM(X_tr, X_vl, X_ts, y_tr, y_vl, y_ts):
    X_tr_ = np.array([np.concatenate((X1,X2),axis=0) for X1, X2 in zip(X_tr, X_vl)])
    X_tr_ = np.concatenate((X_tr_), axis=1)
    y_tr_ = np.concatenate((y_tr, y_vl), axis=0)

    X_ts_ = np.concatenate((X_ts), axis=1)

    clf = SVC(random_state=42)
    parameters = {
        'C': list(10.**np.arange(-10,11)),
        'gamma': list(10.**np.arange(-10,11)),
        'kernel': ['rbf', 'linear'],
        'decision_function_shape': ['ovr', 'ovo']
    }

    gs = GridSearchCV(clf, parameters, n_jobs=8, verbose=1, scoring='accuracy', cv=3)
    gs.fit(X_tr_, y_tr_)

    best_clf = gs.best_estimator_
    print(gs.best_estimator_.get_params())
    best_clf.fit(X_tr_, y_tr_)
    return best_clf.score(X_ts_, y_ts)



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
    parser.add_argument('--settings', metavar='DIR', default='../datasets/settings',
                        help='path to dataset setting files')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    split_path = os.path.join(args.settings, "%s/%s_split%s.txt" % (args.d, "%s", args.s))
    train_path = split_path%("train")
    val_path = split_path%("val")
    test_path = split_path%("test")
    dataset_path = os.path.join(args.settings, "%s/dataset_list.txt" % (args.d))

    npy_data = []
    for npy_path in args.npy_paths:
        npy_data.append(np.load(npy_path))
    
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
    prec = eval(fusion_call)
    print("{:.04f}".format(prec))


    
    

