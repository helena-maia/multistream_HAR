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

sys.path.insert(0, "../../")
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
        vid_name,
        target,
        net,
        num_categories,
        num_samples=25,
        new_size = 299,
        batch_size = 2
        ):

    gc = GradCAM(model=net)

    clip_mean = [0.5]*num_samples
    clip_std = [0.226]*num_samples

    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    val_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize,
        ])

    deep = 1

    # inception = 299,299, resnet = 224,224
    dims = (new_size,new_size,deep,num_samples)
    rgb = np.zeros(shape=dims, dtype=np.float64)
    rgb_flip = np.zeros(shape=dims, dtype=np.float64)
   
    for i in range(num_samples):
        img_file = os.path.join(vid_name, 'vr_{0:02d}.png'.format(i))
        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)   
        rgb[:,:,0,i] = img
        rgb_flip[:,:,0,i] = img[:,::-1]    

    _, _, _, c = rgb.shape
    rgb_list = []
    for c_index in range(c):
        cur_img = rgb[:,:,:,c_index]
        cur_img_tensor = val_transform(cur_img)
        rgb_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
        
    rgb_np = np.concatenate(rgb_list,axis=0)
    prediction = np.zeros((num_categories,rgb.shape[3]))

    index = 50
    input_data = rgb_np[index:index+1,:,:,:]
    imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
    imgDataVar = torch.autograd.Variable(imgDataTensor)

    probs, ids = gc.forward(imgDataVar)
    ids_ = torch.LongTensor([[target]] * len(imgDataVar)).to(torch.device("cuda"))
    gc.backward(ids=ids_)
    regions = gc.generate(target_layer="Mixed_7c")
    save_gradcam(vid_name.split("/")[-1]+".png", gcam=regions[0, 0], raw_image = rgb[:,:,:,index])

    return prediction
