import os,sys
import numpy as np
import cv2
from PIL import Image
from multiprocessing import Pool
import argparse
#from IPython import embed #to debug
#import skvideo.io
import scipy.misc
import random


def ToImg(raw_flow, bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    flow = raw_flow
    flow[flow > bound] = bound
    flow[flow < -bound] = -bound
    flow += bound
    flow *= (255 / float(2 * bound))
    return flow

def save(img, save_dir, num, fmt):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, fmt.format(num))
    ret = cv2.imwrite(save_path, img)

    return ret

def save_flows(flows,save_dir,num,bound):
    '''
    To save the optical flow images and raw images
    :param flows: contains flow_x and flow_y
    :param save_dir: save_dir name (always equal to the video id)
    :param num: the save id
    :param bound: set the bi-bound to flow images
    '''
    #rescale to 0~255 with the bound setting
    flow_x = ToImg(flows[...,0], bound)
    flow_y = ToImg(flows[...,1], bound)

    flow_x_img = flow_x.astype(np.uint8)
    flow_y_img = flow_y.astype(np.uint8)

    ret1 = save(flow_x_img, save_dir, num, 'flow_x_{:05d}.png')
    ret2 = save(flow_y_img, save_dir, num, 'flow_y_{:05d}.png')

    return ret1 and ret2


def extraction(args):
    '''
    Extract dense_flow images using TVL1 and RGB frame(s)
    '''
    video_name, param = args
    dataset_path, output_path, bound, ext, num_rgb, num_of = param

    video_path = os.path.join(dataset_path, video_name + "." + ext)
    img_path = os.path.join(output_path, video_name)
    
    video_capture = cv2.VideoCapture(video_path)
    final_pos = [] # List os selected indices

    if not video_capture.isOpened():
        print('Could not initialize capturing', video_path)
        final_pos = [-1] * (num_rgb + 1) # One position for each RGB frame and one for the OF start
        return final_pos
    
    print(video_path)
    len_video = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    # Extract RGB frame(s)
    def read_save(pos, index):
        fmt = 'img_{:05d}.png'
        ret = False
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, pos)
        flag, frame = video_capture.read()

        if flag:
            ret = save(frame, img_path, index, fmt)
        
        return ret
    
    if num_rgb == 1: # A single RGB frame in the whole video
        rgb_pos = random.randint(0, len_video - 1)
        ret = read_save(rgb_pos, 1)
        final_pos = [rgb_pos] if ret else [-1]
    else: # One RGB frame in each half (num_rgb == 2)
        mid = (len_video - 1) // 2
        rgb_pos_1 = random.randint(0, mid) 
        rgb_pos_2 = random.randint(mid + 1, len_video - 1)

        ret1 = read_save(rgb_pos_1, 1)
        ret2 = read_save(rgb_pos_2, 2)

        final_pos = [rgb_pos_1, rgb_pos_2] if (ret1 and ret2) else [-1 -1]

    # Extract a stack of num_of optical flow (OF) images, starting at a random position
    dtvl1 = cv2.createOptFlow_DualTVL1()
    of_start = random.randint(0, len_video - num_of - 1)
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, of_start)
    frame_count = 0

    flag, frame = video_capture.read()
    while flag and frame_count <= num_of:
        curr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if frame_count > 0:
            flow = dtvl1.calc(prev_frame, curr_frame, None)
            ret = save_flows(flow, img_path, frame_count, bound)

            if not ret: 
                break

        frame_count += 1
        prev_frame = curr_frame
        flag, frame = video_capture.read()

    if frame_count == num_of + 1:
        final_pos.append(of_start)
    else: 
        final_pos.append(-1)

    return final_pos

def parse_args():
    parser = argparse.ArgumentParser(description = "Extract optical flow images using TVL1")
    parser.add_argument('data_root', type = str, help = "Path to dataset videos")
    parser.add_argument('data_list', type = str, help = "List of videos in data_root directory")
    parser.add_argument('output_dir', type = str, help = 'Output path')
    parser.add_argument('position_list', type = str, help = 'Path to the list of selected indices')
    parser.add_argument('--num_workers', default = 4, type = int, help = 'Num of workers to act multi-process (default: 4)')
    parser.add_argument('--bound', default = 20, type = int, help = 'Set the maximum of optical flow (default: 20)')
    parser.add_argument('--v_ext', default = 'avi', type = str, help = 'Video extension (default: avi)')
    parser.add_argument('--n_rgb', default = '2', type = int, help = 'Number of frames to be extracted (default: 2)', choices=[1,2])
    parser.add_argument('--n_flow', default = '10', type = int, help = 'Number of optical flow images to be extracted - stack length (default: 10')
    args = parser.parse_args()
    return args

if __name__ =='__main__':
    args = parse_args()

    data_root = args.data_root
    videos = np.loadtxt(args.data_list, dtype = 'U200', comments = "|")
    output_dir = args.output_dir
    position_list = args.position_list
    num_workers = args.num_workers
    bound = args.bound
    ext = args.v_ext
    n_rgb = args.n_rgb
    n_flow = args.n_flow

    common = [data_root, output_dir, bound, ext, n_rgb, n_flow]

    pool = Pool(num_workers)
    positions = pool.map(extraction, zip(videos, [common]*len(videos)))

    header = ["RGB"] * n_rgb + ["OF"]
    positions.insert(0, header)
    np.savetxt(position_list, positions, fmt="%s")