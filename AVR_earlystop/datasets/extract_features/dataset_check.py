import os
import argparse
import glob
import numpy as np

def getArgs():
    parser = argparse.ArgumentParser(description='Check whether the dataset directory is complete, according to a modality.')
    parser.add_argument("modality_dir", action='store', type=str, help="Modality directory")
    parser.add_argument("dataset_list", action='store', type=str, help="Dataset list with class directory, if applicable")
    parser.add_argument("modality", action='store', choices=["rgb", "flow", "vr"], help="Modality (rgb, flow or vr)")
    parser.add_argument("output_path", action='store', help="Path to the output file with the missing videos")
    parser.add_argument("-m_ext", default="jpg", help="Modality extension (default: jpg)")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = getArgs()

    modality_dir = args.modality_dir
    dataset_list = args.dataset_list
    modality = args.modality
    m_ext = args.m_ext

    videos = np.loadtxt(dataset_list, dtype='U200', comments="|")

    fmts = {'rgb': 'img_00001.%s', 
            'flow': 'flow_[x,y]_00001.%s', 
            'vr': 'visual_rhythm_0000[1,2].%s'}

    fmt = fmts[modality]%m_ext
    num = 1 if modality == 'rgb' else 2
    missing = []

    for i, v in enumerate(videos):
        print(i, len(videos))
        v = glob.escape(v)
        full_path = os.path.join(modality_dir, v, fmt)
        exists = glob.glob(full_path)
        if len(exists) != num:
            missing.append(full_path)

    np.savetxt(args.output_path, missing, fmt="%s")

