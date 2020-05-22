from multiprocessing import Pool
import cv2
import numpy as np
import os
import argparse
import glob

def run(x):
    pass

def getArgs():
    parser = argparse.ArgumentParser(description='Check whether the dataset directory is complete, according to a modality.')
    parser.add_argument("video_dir", action='store', type=str, help="Video directory")
    parser.add_argument("modality_dir", action='store', type=str, help="Modality directory")
    parser.add_argument("modality", action='store', choices=["rgb", "flow", "vr"], help="Modality (rgb, flow or vr)")
    parser.add_argument("-v_ext", default="avi", help="Video extension (default: avi)")
    parser.add_argument("-m_ext", default="jpg", help="Modality extension (default: jpg)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    video_dir = args.dataset_dir
    videos = np.loadtxt(args.video_list, dtype='U200')
    modality = args.modality

    fmts = {'rgb': 'img_%05d.%s', 
            'flow': 'flow_%s_%05d.%s', 
            'vr': 'visual_rhythm_%05d.%s'}

    #if not os.path.isdir(vr_dest):
    #    print("creating folder: "+vr_dest)
    #    os.makedirs(vr_dest)

    
