import math
import csv

def get_list(path, output_path):
    final_list = []

    with open(path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csv_reader)

        video_class = {}

        for i, line in enumerate(csv_reader):
            video_id = line[0]
            action_list = line[9].split(";")

            if action_list[0] != "":
                for j, action in enumerate(action_list):
                    param_list = action.split(" ")
                    class_id = param_list[0]
                    frame_start = math.floor(float(param_list[1])*24) # 24fps
                    frame_end = math.ceil(float(param_list[2])*24)

                    # Check if there is other segments from the same video and class
                    pair = video_id+"_"+class_id
                    if pair in video_class: video_class[pair] += 1
                    else: video_class[pair] = 1

                    segm_ind = "s" + str(video_class[pair])

                    final_list.append([video_id, class_id, frame_start, frame_end, segm_ind])

    with open(output_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=" ")
        csv_writer.writerows(final_list)



train_path = "../Charades/Charades_v1_train.csv"
train_output = "Charades_v1_train_clips.csv"
test_path = "../Charades/Charades_v1_test.csv"
test_output = "Charades_v1_test_clips.csv"

get_list(train_path, train_output)
get_list(test_path, test_output)

