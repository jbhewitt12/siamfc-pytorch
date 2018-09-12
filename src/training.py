import matplotlib.pyplot as plt
import sys
import os
import csv
import numpy as np
from PIL import Image
import time

import src.siamese as siam
from src.visualization import show_frame, show_crops, show_scores

# Steps to train neural net:

# 1. Load and normalize the data 
# "During training, we adopt exemplar images that are 127 ×
# 127 and search images that are 255 × 255 pixels. Images are scaled such that
# the bounding box, plus an added margin for context, has a fixed area."

dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)

# 2. define the CNN

# 3. define the loss function

# 4. Train the network on the training data
