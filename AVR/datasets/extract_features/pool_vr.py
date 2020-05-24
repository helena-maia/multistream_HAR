from multiprocessing import Pool
import cv2
import numpy as np
import os
import argparse
import glob

def run_vr(x):
    ind = x[0]
    video = x[1][0]
    src = x[1][1]
    dest = x[1][2]

    print(ind, video)

    video_path = os.path.join(src, video)
    vr_dest = os.path.join(dest,video[:-4])

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

    hor_vr = np.array([]).reshape(0,int(width),3)
    ver_vr = np.array([]).reshape(0,int(height),3)

    while(cap.isOpened()):
        ret, img = cap.read()

        if ret == True:
            hor = np.mean(img, axis=0)
            ver = np.mean(img, axis=1)
    
            hor_vr = np.vstack([hor_vr,[hor]])
            ver_vr = np.vstack([ver_vr,[ver]])
        else:
            break

    if hor_vr.size == 0 or ver_vr.size == 0: 
        print("Error opening video file ", video_path)
        return

    ver_vr = np.swapaxes(ver_vr, 0,1)
    
    if not os.path.isdir(vr_dest):
        print("creating folder: "+vr_dest)
        os.makedirs(vr_dest)

    hor_vr = cv2.resize(hor_vr, (320,240))
    ver_vr = cv2.resize(ver_vr, (320,240))

    cv2.imwrite(os.path.join(vr_dest,"visual_rhythm_00001.png"), ver_vr)
    cv2.imwrite(os.path.join(vr_dest,"visual_rhythm_00002.png"), hor_vr)

def getArgs():
    parser = argparse.ArgumentParser(description='Compute visual rhythm (mean).')
    parser.add_argument("video_dir", action='store', type=str, help="directory that contains the subclips")
    parser.add_argument("video_list", action='store', type=str, help="list of subclips (without path)")
    parser.add_argument("vr_dest", action='store', type=str, help="directory to save the visual rhythm images")
    parser.add_argument("-ext", type=str, default='avi', help="Video extension (default=avi)")
    parser.add_argument('--num_worker', type=int, default=8, help='')
    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    num_worker = args.num_worker
    video_dir = args.video_dir
    vr_dest = args.vr_dest
    videos = np.loadtxt(args.video_list, dtype='U200', comments="|")
    videos = [v + "." + args.ext for v in videos]

    if not os.path.isdir(vr_dest):
        print("creating folder: "+vr_dest)
        os.makedirs(vr_dest)

    pool = Pool(num_worker)
    pool.map(run_vr,enumerate(zip(videos, len(videos)*[video_dir], len(videos)*[vr_dest])))
