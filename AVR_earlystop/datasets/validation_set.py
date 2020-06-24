import argparse
import os
import sys
from sklearn.model_selection import train_test_split
import numpy as np

def make_dataset(source):
    '''
        Method to obtain the path, duration(number of frame) and target of each
        video of the dataset from a file(train or test) that contain this detalls
        in each line.
    '''
    if not os.path.exists(source):
        print('Setting file %s for the dataset doesnt exist.' % (source))
        sys.exit()
    else:
        clips = []
        with open(source) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split()
                clip_path = line_info[0]
                duration = int(line_info[1])
                target = int(line_info[2])
                item = (clip_path, duration, target)
                clips.append(item)
    return clips


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--split", "-s", type=int, default=1,
                        help='')
    parser.add_argument("--dataset", "-d", type=str, default='ucf101', choices=['ucf101','hmdb51'],
                        help='')
    parser.add_argument("--settings", "-st", type=str, default='settings/',
                        help='')

    args = parser.parse_args()

    train_file = os.path.join(args.settings, args.dataset, "train_split%d.txt"%args.split)
    clips = make_dataset(train_file)
    y = [c[2] for c in clips]
    
    clips_train, clips_val, _, _ = train_test_split(clips, y, test_size=0.2, random_state=42, stratify=y)

    clips_train = sorted(clips_train)
    clips_val = sorted(clips_val)

    new_val_file = os.path.join(args.settings, args.dataset, "val_split%d.txt"%args.split)
    np.savetxt(train_file, clips_train, fmt="%s") # Replace train file
    np.savetxt(new_val_file, clips_val, fmt="%s") # Replace train file




    