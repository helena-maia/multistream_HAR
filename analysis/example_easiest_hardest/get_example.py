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

    return name_list, label_list, file_lines

def get_example(dataset, settings, npy_path_s1, class_index, correct):
    num_classes = 101 if dataset == "ucf101" else 51

    test_path = os.path.join(settings, "%s/test_split1.txt" % (dataset))
    _, ts_labels, file_lines = get_labels(test_path)
    npy_data = np.load(npy_path_s1)

    example = -1

    for i, data in enumerate(npy_data):
        if ts_labels[i] == class_index:
            y_pred = np.argmax(data)
            if correct == (y_pred == ts_labels[i]):
                example = i
                break

    return file_lines[example]
    
