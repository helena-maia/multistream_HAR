import os
import glob
import cv2
from multiprocessing import Pool
import argparse
import numpy as np
import itertools

def getArgs():
    parser = argparse.ArgumentParser(description='Extracts statistics information from optical flow images from ucf or hmdb datasets')
    parser.add_argument("dataset_path", action='store', type=str, help="Path to the main directory that contains the optical flow images. This dataset directory must include either the videos directories (ucf101: dataset_path/video_dir/flow_images) or the classes directories (hmdb: dataset_path/class_dir/video_dir/flow_images).")
    parser.add_argument('--num_worker', type=int, default=8, help='')
    return parser.parse_args()

def get_statistics(video_path):
	flow_x_list = glob.glob(os.path.join(video_path,"flow_x_*"))
	flow_y_list = glob.glob(os.path.join(video_path,"flow_y_*"))
	accum_mean_x = 0.
	accum_std_x = 0.
	count_low_std_x_1 = 0.
	count_low_std_x_5 = 0.
	count_low_std_x_10 = 0.
	accum_mean_y = 0.
	accum_std_y = 0.
	count_low_std_y_1 = 0.
	count_low_std_y_5 = 0.
	count_low_std_y_10 = 0.

	for pair in zip(flow_x_list, flow_y_list):
		flow_x_img = cv2.imread(pair[0], 0)
		flow_y_img = cv2.imread(pair[1], 0)

		mean_x = flow_x_img.mean()
		std_x = flow_x_img.std()
		mean_y = flow_y_img.mean()
		std_y = flow_y_img.std()


		count_low_std_x_1 += 1 if std_x <= 1. else 0
		count_low_std_y_1 += 1 if std_y <= 1. else 0

		count_low_std_x_5 += 1 if std_x <= 5. else 0
		count_low_std_y_5 += 1 if std_y <= 5. else 0

		count_low_std_x_10 += 1 if std_x <= 10. else 0
		count_low_std_y_10 += 1 if std_y <= 10. else 0

		accum_mean_x += mean_x
		accum_std_x += std_x
		accum_mean_y += mean_y
		accum_std_y += std_y

	if len(flow_x_list):
		accum_mean_x /= len(flow_x_list)
		accum_mean_y /= len(flow_x_list)
		accum_std_x /= len(flow_x_list)
		accum_std_y /= len(flow_x_list)
		count_low_std_x_1 /= len(flow_x_list)
		count_low_std_x_5 /= len(flow_x_list)
		count_low_std_x_10 /= len(flow_x_list)
		count_low_std_y_1 /= len(flow_x_list)
		count_low_std_y_5 /= len(flow_x_list)
		count_low_std_y_10 /= len(flow_x_list)

	return [accum_mean_x, accum_std_x, count_low_std_x_1, count_low_std_x_5, count_low_std_x_10, 
			accum_mean_y, accum_std_y, count_low_std_y_1, count_low_std_y_5, count_low_std_y_10]

def divide_into_classes(video_list):
	get_class = lambda video_path: video_path.split("/")[-1].split("_")[1]
	
	videos_dict = {k:list(v) for k, v in itertools.groupby(video_list, key = get_class)}

	return videos_dict

if __name__ == "__main__":
	args = getArgs()
	dataset_path = args.dataset_path
	second_level = glob.glob(os.path.join(dataset_path, "*"))
	pool = Pool(args.num_worker)

	# Check directory organization
	third_level = glob.glob(os.path.join(second_level[0], "flow_x_00001.jpg"))
	if len(third_level) == 0: # Second level refers to classes (hmdb)
		for c in second_level:
			videos = sorted(glob.glob(os.path.join(c, "*")))
			statistics = pool.map(get_statistics,videos)
			statistics_np = np.array(statistics)
			class_name = os.path.basename(c)
			np.savetxt(class_name + ".csv", statistics_np, fmt="%s", delimiter=" ")
	else: # Second level refers to videos (ucf)
		videos = sorted(second_level)
		videos_dict = divide_into_classes(videos)

		for c in videos_dict:
			statistics = pool.map(get_statistics, videos_dict[c])
			statistics_np = np.array(statistics)
			np.savetxt(c + ".csv", statistics_np, fmt="%s", delimiter=" ")

			
		
		

