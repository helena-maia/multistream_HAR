#!/usr/bin/env python

import os
import sys
import argparse
import collections
import numpy as np
import cv2
import math
import random
import time
import json

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

sys.path.insert(0, "../../../AVR_earlystop/")
import models
from VideoSpatialPrediction import VideoSpatialPrediction

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

parser = argparse.ArgumentParser(description='PyTorch Two-Stream Action Recognition - Test')
parser.add_argument('--modality', '-m', metavar='MODALITY', default='rgb',
                    choices=["rgb", "rhythm"],
                    help='modality: rgb | rhythm ')
parser.add_argument('--dataset', '-d', metavar='MODALITY', default='ucf101',
                    choices=["ucf101", "hmdb51"],
                    help='modality: ucf101 | hmdb51 ')
parser.add_argument('-s', '--split', default=1, type=int, metavar='S',
                    help='which split of data to work on (default: 1)')
parser.add_argument('--architecture', '-a', metavar='MODALITY', default='inception_v3',
                    choices=["resnet152", "inception_v3"])
parser.add_argument('--vr_approach', '-vra', default=3, type=int,
                    metavar='N', help='visual rhythm approach (choices: 1 - vertical, 2 - horizontal, 3 - AVR per class, 4 - AVR per video)(default: 3)')
parser.add_argument('--settings', metavar='DIR', default='../datasets/settings',
                    help='path to dataset setting files (default: ../datasets/settings)')
parser.add_argument('-w', action='store_true', help="Compute features for the whole dataset, if the flag is found")
parser.add_argument('-o', metavar='DIR', default='NPYS/',
                    help='path to save npy and log files (default: NPYS/)')
parser.add_argument('data_dir')
parser.add_argument('model_path')

def softmax(x):
    y = [math.exp(k) for k in x]
    sum_y = math.fsum(y)
    z = [k/sum_y for k in y]

    return z

def logging(args):
    args_dict = vars(args)
    args_dict['hostname'] = os.uname()[1]

    if 'VIRTUAL_ENV' in os.environ.keys(): 
        args_dict['virtual_env'] = os.environ['VIRTUAL_ENV']
    else: 
        print("WARNING: No virtualenv activated")
        args_dict['virtual_env'] = None

    timestamp = time.time() 
    full_path = os.path.join(args.o, str(timestamp))
    if not os.path.isdir(full_path):
        os.makedirs(full_path)

    log_path = os.path.join(full_path, "args.json")
    with open(log_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4, sort_keys=True)

    os.system('pip freeze > ' + os.path.join(full_path,'requirements.txt'))

    print("Saving everything to directory %s." % (full_path))

    return full_path

def main():
    args = parser.parse_args()
    output_path = logging(args)
    model_path = args.model_path
    data_dir = args.data_dir
    
    start_frame = 0
    num_samples = 25 if args.modality[:3]=='rgb' else 1
    ext = ".png" if args.modality == "rhythm" and args.dataset == "hmdb51" else ".jpg"
    num_categories = 51 if args.dataset=='hmdb51' else 101
    new_size = 224
    
    model_start_time = time.time()
    params = torch.load(model_path)

    if args.architecture == "inception_v3":
        new_size=299
        if args.modality == "rhythm":
            spatial_net = models.flow_inception_v3(pretrained=False, channels = 1, num_classes=num_categories)
        else:
            spatial_net = models.rgb_inception_v3(pretrained=False, channels = 3, num_classes=num_categories)
    else:
        if args.modality == "rhythm":
            spatial_net = models.flow_resnet152(pretrained=False, channels = 1, num_classes=num_categories)
        else:
            spatial_net = models.rgb_resnet152(pretrained=False, channels = 3, num_classes=num_categories)

    spatial_net.load_state_dict(params['state_dict'])
    spatial_net.cuda()
    spatial_net.eval()
    model_end_time = time.time()
    model_time = model_end_time - model_start_time
    print("Action recognition spatial model is loaded in %4.4f seconds." % (model_time))

    test_path = os.path.join(args.settings, args.dataset)
    test_file = os.path.join(test_path, "dataset_list.txt") if args.w else os.path.join(test_path, "test_split%d.txt"%(args.split))
    print(test_file)
    f_test = open(test_file, "r")
    test_list = f_test.readlines()
    print("we got %d videos" % len(test_list))

    line_id = 1
    match_count = 0
    result_list = []
    
    lines = []
    if args.vr_approach == 3:
        direction_path = os.path.join(args.settings, args.dataset, "direction.txt")
        lines = [int(line.rstrip('\n')) for line in open(direction_path)]
    elif args.vr_approach == 4:
        direction_path = os.path.join(args.settings, args.dataset, "direction_video.txt")
        lines = {line.split()[0]: int(line.split()[1]) for line in open(direction_path)}

    for line in test_list:
        line_info = line.split(" ")
        clip_path = os.path.join(data_dir, line_info[0])
        num_frames = int(line_info[1])
        input_video_label = int(line_info[2])

        video_name = clip_path.split("/")[-1]
        index = lines[input_video_label] if args.vr_approach == 3 else (lines[video_name] if args.vr_approach == 4 else args.vr_approach)

        spatial_prediction = VideoSpatialPrediction(
                args.modality,
                clip_path,
                input_video_label,
                spatial_net,
                num_categories,
                start_frame,
                num_frames,
                num_samples,
                index,
                new_size = new_size,
                ext = ext)
        avg_spatial_pred_fc8 = np.mean(spatial_prediction, axis=1)
        result_list.append(avg_spatial_pred_fc8)

        pred_index = np.argmax(avg_spatial_pred_fc8)
        print(args.modality+" split "+str(args.split)+", sample %d/%d: GT: %d, Prediction: %d ==> correct: %d" % 
            (line_id, len(test_list), input_video_label, pred_index, match_count))

        if pred_index == input_video_label:
            match_count += 1
        line_id += 1

    print(match_count)
    print(len(test_list))
    print("Accuracy is: %4.4f" % (float(match_count)/len(test_list)))

    #npy_name = args.dataset+"_"+args.modality+"_"+args.architecture+"_s"+str(args.split)+".npy"
    #npy_path = os.path.join(output_path, npy_name)
    #np.save(npy_path, np.array(result_list))

if __name__ == "__main__":
    main()

