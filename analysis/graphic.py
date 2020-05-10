import os
import glob
import cv2
from multiprocessing import Pool
import argparse
import numpy as np
import matplotlib.pyplot as plt

def getArgs():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument("csv_path", action='store', type=str, help="path to one CSV file or a directory with multiple CSV files", default=None)
    return parser.parse_args()

threshold = 10.0

def graphic(statistics_np):
	fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
		
	y = statistics_np[:,0]
	yerr = statistics_np[:,1]
	x = np.arange(y.shape[0])
	ax = axs[0,0]
	ax.errorbar(x=x,y=y,yerr=yerr, fmt='o')
	ax.set_title("Intensity average - X")

	c1 = statistics_np[:,2]
	c5 = statistics_np[:,3]
	c10 = statistics_np[:,4]
	ax = axs[1,0]
	ax.plot(c1,marker='o',linestyle="None")
	ax.plot(c5,marker='o',linestyle="None")
	ax.plot(c10,marker='o',linestyle="None")
	ax.set_title("Ratio lower dev - X")

	y = statistics_np[:,5]
	yerr = statistics_np[:,6]
	x = np.arange(y.shape[0])
	ax = axs[0,1]
	ax.errorbar(x=x,y=y,yerr=yerr, fmt='o')
	ax.set_title("Intensity average - Y")

	c1 = statistics_np[:,7]
	c5 = statistics_np[:,8]
	c10 = statistics_np[:,9]
	ax = axs[1,1]
	ax.plot(c1, marker='o',linestyle="None")
	ax.plot(c5, marker='o',linestyle="None")
	ax.plot(c10, marker='o',linestyle="None")
	ax.set_title("Ratio lower dev - Y")

	plt.show()

if __name__ == "__main__":
	args = getArgs()
	csv_path = args.csv_path

	if os.path.isdir(csv_path):
		csv_list = sorted(glob.glob(os.path.join(csv_path,"*.csv")))
		mean_list = []

		for csv_file in csv_list:
			print(csv_file)
			statistics_np = np.loadtxt(csv_file, dtype=float, delimiter=" ")
			statistics_mean = np.mean(statistics_np, axis=0)
			mean_list.append(statistics_mean)

		mean_np = np.array(mean_list)

		graphic(mean_np)

	else:
		statistics_np = np.loadtxt(csv_path, dtype=float, delimiter=" ")

		graphic(statistics_np)
		

