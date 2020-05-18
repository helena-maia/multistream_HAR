# Multi-Stream Convolutional Neural Network for Action Recognition Based on Adaptive Visual Rhythm

Based on the [Adaptive-Visual-Rhythms-for-Action-Recognition](https://github.com/darwinTC/Adaptive-Visual-Rhythms-for-Action-Recognition) and [Two Stream Pytorch](https://github.com/bryanyzhu/two-stream-pytorch).

python main_single_gpu.py DATA_PATH -m rgb2 -a rgb_inception_v3 --new_length=1 --epochs 250 --lr 0.001 --lr_steps 100 200

python main_single_gpu.py DATA_PATH -m rhythm -a flow_inception_v3 --new_length=1 --epochs 350 --lr 0.001 --lr_steps 200 300

python main_single_gpu.py DATA_PATH -m flow -a flow_inception_v3 --new_length=10 --epochs 350 --lr 0.001 --lr_steps 200 300
