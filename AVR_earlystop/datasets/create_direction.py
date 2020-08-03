import os
#import sys
import glob
import argparse
import numpy as np
import cv2
#from multiprocessing import Pool, current_process
#from skimage.feature import hog
#from skimage import data, color, exposure
#from itertools import product
list_class = []
map_class ={} # Video name as key, video class as value
video_mov = {} # Predominant direction of each video 2 = horizontal, 1 = vertical

def run(ind, video):
    '''
    Determines the predominant direction of movement of a given video.
    Based on https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
    '''

    path_split = video.split("/")
    vid_name = path_split[-1].split('.')[0] # Video name without extension
    vid_class = path_split[-2]

    # Map each video with its respective video class
    map_class[vid_name]=vid_class

    cap = cv2.VideoCapture(video)

    # Parameters for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    accum_mov_x = 1 # Accumulated horizontal movement
    accum_mov_y = 0 # Accumulated vertical movement

    while True:
        ret,frame = cap.read()

        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        except cv2.error:
            break

        if (p1 is None) or (p0 is None):
            break

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Accumulate displacements
        for i, (new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            accum_mov_x += abs(a-c)
            accum_mov_y += abs(b-d)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

    # save b each video its respective direction (1=ejey, 2=ejex)
    if accum_mov_x > accum_mov_y:
        video_mov[vid_name] = 2
        dir_ = 'x'
    else:
        video_mov[vid_name] = 1
        dir_ = 'y'

    print(str(ind)+' : name video: '+vid_name+' => direction '+dir_)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Determine the direction of movement by class of a given dataset')
    parser.add_argument('src_dir', type=str, help='path to the video data (structure: src_dir/class/video)')
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--ext', type=str, default='avi', choices=['avi','mp4'],
                        help='video file extensions')
    parser.add_argument('--o', type=str, default='direction.txt', help='output: path to direction file')
    parser.add_argument('-g', action='store_true', help='group by class')

    args = parser.parse_args()
    src_dir = args.src_dir
    num_worker = args.num_worker
    ext = args.ext
    output = args.o
    grouped = args.g

    full_path = os.path.join(src_dir, "*/*."+ext)
    video_list = glob.glob(full_path)
    print(str(len(video_list))+" videos were found")

    for i, vid in enumerate(video_list):
        run(i, vid)

    if args.g:
        dic_final = {}
        list_name_vid = map_class.keys()
        list_class = map_class.values()

        for vid in list_name_vid:
            if first_data[vid] in dic_final:
                if second_data[vid]==2:
                    dic_final[first_data[vid]][0]+=1
                else:
                    dic_final[first_data[vid]][1]+=1        
            else:
                if second_data[vid]==2:
                    dic_final[first_data[vid]]=[1,0]
                else:
                    dic_final[first_data[vid]]=[0,1]
         direction = []
    for i,vid in enumerate(list_class):
        idx = '2' if dic_final[vid][0]>dic_final[vid][1] else '1'
        direction.append(idx)

    open('directionHMDB.txt', 'w').writelines(direction)
    '''
