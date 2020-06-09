import csv
import os
import numpy as np
import shutil

def copy_video(info):
    video_id, class_id, frame_start, frame_end, segm_ind = info[:5]
    src_img_fmt, src_flow_fmt, dst_dir, dst_img_fmt, dst_flow_fmt = info[5:]
    frame_start, frame_end = int(frame_start) + 1, int(frame_end) + 1
    dst_dir = dst_dir_fmt%(class_id, video_id, class_id, segm_ind)

    if not os.path.isdir(dst_dir):
        print("Creating video directory: ", dst_dir)
        os.makedirs(dst_dir)

    count = 1

    for i in range(frame_start, frame_end + 1):
        src_img = src_img_fmt%(video_id, video_id, j)
        src_flow_x = src_flow_fmt%(video_id, video_id, j, "x")
        src_flow_y = src_flow_fmt%(video_id, video_id, j, "y")
        if not (os.path.isfile(src_img) and os.path.isfile(src_flow_x) and os.path.isfile(src_flow_y)):
            continue

        dst_img = os.path.join(dst_dir, dst_img_fmt%count)
        dst_flow_x = os.path.join(dst_dir, dst_flow_fmt%("x", count))
        dst_flow_y = os.path.join(dst_dir, dst_flow_fmt%("y", count))

        shutil.copy(src_img, dst_img)
        shutil.copy(src_flow_x, dst_flow_x)
        shutil.copy(src_flow_y, dst_flow_y)

        #print(src_img, dst_img)
        #print(src_flow_x, dst_flow_x)
        #print(src_flow_y, dst_flow_y)

        count += 1



def crop(list_path, src_path, dst_path):
    final_list = []

    with open(list_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')

        src_img_fmt = os.path.join(src_path[0], "%s/%s-%06d.jpg")
        src_flow_fmt = os.path.join(src_path[1], "%s/%s-%06d%s.jpg")

        dst_dir_fmt = os.path.join(dst_path, "%s/%s_%s_%s/")
        dst_img_fmt = "img_%05d.jpg"
        dst_flow_fmt = "flow_%s_%05d.jpg"

        for i, line in enumerate(csv_reader):
            video_id, class_id, frame_start, frame_end, segm_ind = line
            frame_start, frame_end = int(frame_start) + 1, int(frame_end) + 1
            dst_dir = dst_dir_fmt%(class_id, video_id, class_id, segm_ind)

            if not os.path.isdir(dst_dir):
                print("Creating video directory: ", dst_dir)
                os.makedirs(dst_dir)

            count = 1
            for j in range(frame_start, frame_end+1):
                src_img = src_img_fmt%(video_id, video_id, j)
                src_flow_x = src_flow_fmt%(video_id, video_id, j, "x")
                src_flow_y = src_flow_fmt%(video_id, video_id, j, "y")
                if not (os.path.isfile(src_img) and os.path.isfile(src_flow_x) and os.path.isfile(src_flow_y)):
                    continue

                dst_img = os.path.join(dst_dir, dst_img_fmt%count)
                dst_flow_x = os.path.join(dst_dir, dst_flow_fmt%("x", count))
                dst_flow_y = os.path.join(dst_dir, dst_flow_fmt%("y", count))

                shutil.copy(src_img, dst_img)
                shutil.copy(src_flow_x, dst_flow_x)
                shutil.copy(src_flow_y, dst_flow_y)

                #print(src_img, dst_img)
                #print(src_flow_x, dst_flow_x)
                #print(src_flow_y, dst_flow_y)

                count += 1

            relative_path = "%s/%s_%s_%s"%(class_id, video_id, class_id, segm_ind)
            label = int(class_id[1:])
            final_list.append([relative_path, count, label])

    return final_list


src_path = ["/home/helena.maia/Documentos/Datasets/Charades_v1_rgb/",
            "/home/helena.maia/Documentos/Datasets/Charades_v1_flow/"]
train_subclips = "Charades_v1_train_clips.csv"
test_subclips = "Charades_v1_test_clips.csv"
dst_path = "/home/helena.maia/Documentos/Datasets/Charades_v1_subclips/"

if not os.path.isdir(dst_path):
    print("Creating output directory: ", dst_path)
    os.makedirs(dst_path)

train_list = sorted(crop(train_subclips, src_path, dst_path))
test_list = sorted(crop(test_subclips, src_path, dst_path))

np.savetxt("train.txt", train_list, fmt="%s")
np.savetxt("test.txt", test_list, fmt="%s")

