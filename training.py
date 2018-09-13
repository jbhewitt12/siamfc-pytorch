import matplotlib.pyplot as plt
import sys
import os
import csv
import numpy as np
from PIL import Image
import time

from src.tracker import tracker
from src.parse_arguments import parse_arguments
from src.region_to_bbox import region_to_bbox
from src.siamese import SiameseNet
from src.visualization import show_frame, show_crops, show_scores
from run_tracker_evaluation import _init_video

# Steps to train neural net:

# -->> 1. Load and normalize the data 
# "During training, we adopt exemplar images that are 127 ×
# 127 and search images that are 255 × 255 pixels. Images are scaled such that
# the bounding box, plus an added margin for context, has a fixed area."

hp, evaluation, run, env, design = parse_arguments()

# evaluation.dataset = 'vot2013'

dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
# dataset_folder = 'C:/Users/Josh/Documents/Uni/Capstone_B/PyTorch/siamfc-pytorch/data/vot2013'
videos_list = [v for v in os.listdir(dataset_folder) if not v[0] == '.']
videos_list.sort()
print(videos_list)
gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_list[0])

##from siam.get_scores
# image = Image.open(filename)
# avg_chan = ImageStat.Stat(image).mean
# frame_padded_x, npad_x = pad_frame(image, image.size, pos_x, pos_y, scaled_search_area[2], avg_chan)
# x_crops = extract_crops_x(frame_padded_x, npad_x, pos_x, pos_y, scaled_search_area[0], scaled_search_area[1], scaled_search_area[2], design.search_sz)

# def show_crops(crops, fig_n)

# -->> 2. define the CNN

# -->> 3. define the loss function

# -->> 4. Train the network on the training data
