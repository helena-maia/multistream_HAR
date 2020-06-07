import csv
import os

def crop(list_path, src_path, dst_path):
    with open(list_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=' ')
        src_fmt = os.path.join(src_path, "%s/%s_%06d.jpg")
        dst_dir_fmt = os.path.join(dst_path, "%s/%s_%s_%s/")
        dst_img_fmt = "img_%05d.jpg"

        for i, line in enumerate(csv_reader):
            video_id, class_id, frame_start, frame_end, segm_ind = line
            frame_start, frame_end = int(frame_start), int(frame_end)
            dst_dir = dst_dir_fmt%(class_id, video_id, class_id, segm_ind)
            print(video_id, class_id, frame_start, frame_end, segm_ind)

            if not os.path.isdir(dst_dir):
                print("Creating video directory: ", dst_dir)
                os.makedirs(dst_dir)

            count = 1
            for j in range(frame_start, frame_end+1):
                src_img = src_fmt%(video_id, video_id, j)
                dst_img = os.path.join(dst_dir, dst_img_fmt%count)
                shutil.copy(src_img, dst_img)
                count += 1


src_path = "/home/helena.maia/Documentos/Datasets/Charades_v1_rgb/"
train_subclips = "Charades_v1_train_clips.csv"
test_subclips = "Charades_v1_test_clips.csv"
dst_path = "/home/helena.maia/Documentos/Datasets/Charades_v1_rgb_subclips/"

if not os.path.isdir(dst_path):
    print("Creating output directory: ", dst_path)
    os.makedirs(dst_path)

crop(train_subclips, src_path, dst_path)
crop(test_subclips, src_path, dst_path)

