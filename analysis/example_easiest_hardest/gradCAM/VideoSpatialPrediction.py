'''
A sample function for classification using spatial network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import os
import sys
import numpy as np
import math
import cv2
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

sys.path.insert(0, "../")
import video_transforms

sys.path.insert(0, "grad-cam-pytorch-master/")
from grad_cam import GradCAM
import matplotlib.cm as cm

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite("img_"+filename, raw_image)
    cv2.imwrite(filename, np.uint8(gcam))

def VideoSpatialPrediction(
        mode,
        vid_name,
        target,
        net,
        num_categories,
        start_frame=0,
        num_frames=0,
        num_samples=25,
        index =1,
        new_size = 299,
        ext = ".jpg"
        ):

    gc = GradCAM(model=net)

    if num_frames == 0:
        imglist = os.listdir(vid_name)
        duration = len(imglist)
    else:
        duration = num_frames

    # selection
    if mode == 'rgb':
        step = int(math.floor((duration-1)/(num_samples-1)))
        clip_mean = [0.485, 0.456, 0.406]
        clip_std = [0.229, 0.224, 0.225]
    else:
        clip_mean = [0.5, 0.5]
        clip_std = [0.226, 0.226]

    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    test_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    # inception = 320,360, resnet = 240, 320
    width = 320 if new_size==299 else 240
    height = 360 if new_size==299 else 320
    deep = 1 if mode == 'rhythm' else 3
    dims = (width,height,deep,num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        if mode == 'rhythm':
            img_file = os.path.join(vid_name, 'visual_rhythm_{0:05d}{1}'.format(index, ext))
            #print(img_file)
            img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)   
            img = cv2.resize(img, dims[1::-1])
            rgb[:,:,0,i] = img
            rgb_flip[:,:,0,i] = img[:,::-1]
        else:
            img_file = os.path.join(vid_name, 'img_{0:05d}{1}'.format(i*step+1, ext))
            img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, dims[1::-1])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb[:,:,:,i] = img
            rgb_flip[:,:,:,i] = img[:,::-1,:]


    # crop 299 = inception, 224 = resnet
    size = new_size
    corner = [(height-size)//2, (width-size)//2]
    rgb_1 = rgb[:size, :size, :,:]
    rgb_2 = rgb[:size, -size:, :,:]
    rgb_3 = rgb[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
    rgb_4 = rgb[-size:, :size, :,:]
    rgb_5 = rgb[-size:, -size:, :,:]
    rgb_f_1 = rgb_flip[:size, :size, :,:]
    rgb_f_2 = rgb_flip[:size, -size:, :,:]
    rgb_f_3 = rgb_flip[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
    rgb_f_4 = rgb_flip[-size:, :size, :,:]
    rgb_f_5 = rgb_flip[-size:, -size:, :,:]

    rgb = np.concatenate((rgb_1,rgb_2,rgb_3,rgb_4,rgb_5,rgb_f_1,rgb_f_2,rgb_f_3,rgb_f_4,rgb_f_5), axis=3)

    _, _, _, c = rgb.shape
    rgb_list = []
    for c_index in range(c):
        cur_img = rgb[:,:,:,c_index]
        cur_img_tensor = test_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))

    rgb_np = np.concatenate(rgb_list,axis=0)
    prediction = np.zeros((num_categories,rgb.shape[3]))

    index = 10
    index2 = index*10
    print(rgb_1.shape, rgb_np.shape)
    input_data = rgb_np[index2:index2+1,:,:,:]
    imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
    imgDataVar = torch.autograd.Variable(imgDataTensor)

    probs, ids = gc.forward(imgDataVar)
    ids_ = torch.LongTensor([[target]] * len(imgDataVar)).to(torch.device("cuda"))
    gc.backward(ids=ids_)
    regions = gc.generate(target_layer="Mixed_7c")
    save_gradcam(vid_name.split("/")[-1]+".png", gcam=regions[0, 0], raw_image = rgb_1[:,:,:,index])

    return prediction
