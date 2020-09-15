'''
A sample function for classification using temporal network
Customize as needed:
e.g. num_categories, layer for feature extraction, batch_size
'''

import glob
import os
import sys
import numpy as np
import math
import cv2

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

def VideoTemporalPrediction(
        mode,
        vid_name,
        target,
        net,
        num_categories,
        start_frame=0,
        num_frames=0,
        num_samples=25,
        optical_flow_frames=10,
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
    step = int(math.floor((duration-optical_flow_frames+1)/num_samples))
    clip_mean = [0.5] * optical_flow_frames * 2
    clip_std = [0.226] * optical_flow_frames * 2

    normalize = video_transforms.Normalize(mean=clip_mean,
                                     std=clip_std)
    test_transform = video_transforms.Compose([
            video_transforms.ToTensor(),
            normalize
        ])

    
    # inception = 320,360, resnet = 240, 320
    width = 320 if new_size==299 else 240
    height = 360 if new_size==299 else 320
    deep = optical_flow_frames*2
    dims = (width,height,deep,num_samples)
    flow = np.zeros(shape=dims, dtype=np.float64)
    flow_flip = np.zeros(shape=dims, dtype=np.float64)

    for i in range(num_samples):
        for j in range(optical_flow_frames):
            flow_x_file = os.path.join(vid_name, mode+'_x_{0:05d}{1}'.format(i*step+j+1 + start_frame, ext))
            flow_y_file = os.path.join(vid_name, mode+'_y_{0:05d}{1}'.format(i*step+j+1 + start_frame, ext))
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            img_x = cv2.resize(img_x, dims[1::-1])
            img_y = cv2.resize(img_y, dims[1::-1])

            flow[:,:,j*2  ,i] = img_x
            flow[:,:,j*2+1,i] = img_y

            flow_flip[:,:,j*2  ,i] = 255 - img_x[:, ::-1]
            flow_flip[:,:,j*2+1,i] = img_y[:, ::-1]

    # crop 299 = inception, 224 = resnet
    size = new_size
    corner = [(height-size)//2, (width-size)//2]
    flow_1 = flow[:size, :size, :,:]
    flow_2 = flow[:size, -size:, :,:]
    flow_3 = flow[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
    flow_4 = flow[-size:, :size, :,:]
    flow_5 = flow[-size:, -size:, :,:]
    flow_f_1 = flow_flip[:size, :size, :,:]
    flow_f_2 = flow_flip[:size, -size:, :,:]
    flow_f_3 = flow_flip[corner[1]:corner[1]+size, corner[0]:corner[0]+size, :,:]
    flow_f_4 = flow_flip[-size:, :size, :,:]
    flow_f_5 = flow_flip[-size:, -size:, :,:]

    flow = np.concatenate((flow_1,flow_2,flow_3,flow_4,flow_5,flow_f_1,flow_f_2,flow_f_3,flow_f_4,flow_f_5), axis=3)
    
    _, _, _, c = flow.shape
    flow_list = []
    for c_index in range(c):
        cur_img = flow[:,:,:,c_index].squeeze()
        cur_img_tensor = test_transform(cur_img)
        flow_list.append(np.expand_dims(cur_img_tensor.numpy(), 0))
        
    flow_np = np.concatenate(flow_list,axis=0)
    prediction = np.zeros((num_categories,flow.shape[3]))

    index = 50
    input_data = flow_np[index:index+1,:,:,:]
    raw_image_x = flow[:,:,[0,2,4,6,8],index]
    raw_image_y = flow[:,:,[1,3,5,7,9],index]
    print(raw_image_x.shape, raw_image_y.shape)
    imgDataTensor = torch.from_numpy(input_data).type(torch.FloatTensor).cuda()
    imgDataVar = torch.autograd.Variable(imgDataTensor)

    probs, ids = gc.forward(imgDataVar)
    ids_ = torch.LongTensor([[target]] * len(imgDataVar)).to(torch.device("cuda"))
    gc.backward(ids=ids_)
    regions = gc.generate(target_layer="Mixed_7c")
    save_gradcam(vid_name.split("/")[-1]+"_x.png", gcam=regions[0, 0], raw_image = flow[:,:,4:5,index])
    save_gradcam(vid_name.split("/")[-1]+"_y.png", gcam=regions[0, 0], raw_image = flow[:,:,5:6,index])

    return prediction
