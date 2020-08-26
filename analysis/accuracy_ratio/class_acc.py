import argparse
import os
import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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

def obtain_accuracies_per_class(data_complete, ground_truth, num_classes=101):
    class_count = np.zeros(num_classes)
    y_pred = np.zeros(len(data_complete))

    for i, data in enumerate(data_complete):
        index = np.argmax(data)
        y_pred[i] = index
        class_count[ground_truth[i]] += 1

    conf_matrix = confusion_matrix(ground_truth, y_pred)
    match_count = np.diagonal(conf_matrix)

    return match_count / class_count, conf_matrix

def class_acc(dataset, settings, npy_paths):
    num_classes = 101 if dataset == "ucf101" else 51
    accum_acc = np.zeros(num_classes, dtype=float)

    for s in range(1,4):
        test_path = os.path.join(settings, "%s/test_split%s.txt" % (dataset, s))
        _, ts_labels = get_labels(test_path)
        npy_data = np.load(npy_paths[s-1])
        acc_class, _ = obtain_accuracies_per_class(npy_data, ts_labels, num_classes)

        accum_acc += acc_class

    accum_acc /= 3

    return accum_acc
    
