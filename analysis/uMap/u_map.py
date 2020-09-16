import os
import cv2
import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

test_file = "ucf101_test_split1.txt"
image_dir = "inception_vr_ucf101_VR2_resize_inception"

f_test = open(test_file, "r")
test_list = f_test.readlines()

flatten_imgs = []
labels = []

for i, line in enumerate(test_list):
    if i > 1000: break
    print(i)
    line_info = line.split(" ")
    clip_path = os.path.join(image_dir, line_info[0])
    num_frames = int(line_info[1])
    clip_label = int(line_info[2])
    
    img_path = os.path.join(clip_path, "vr_00.png")
    if os.path.isfile(img_path):
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        flatten_imgs.append(gray_img.flatten())
        labels.append(clip_label)

reducer = umap.UMAP()
scaled_flatten_imgs = StandardScaler().fit_transform(flatten_imgs)
embedding = reducer.fit_transform(scaled_flatten_imgs)

num_classes = np.unique(labels).size

embedding = np.array(embedding)
labels = np.array(labels)

for c in range(num_classes):
    data = embedding[labels==c]
    print(embedding.shape, data.shape)
    plt.scatter(
        data[:, 0],
        data[:, 1],
        marker=".")

plt.savefig("fig.png")
