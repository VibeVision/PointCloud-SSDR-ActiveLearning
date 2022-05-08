from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, glob, pickle
import sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

grid_size = 0.06
dataset_path = './data/semantic3d/original_data'
original_pc_folder = join(dataset_path, 'original_ply')
sub_pc_folder = join(dataset_path, 'input_{:.3f}'.format(grid_size))
os.mkdir(original_pc_folder) if not exists(original_pc_folder) else None
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None
cloud_name_list = ['bildstein_station1_xyz_intensity_rgb',
                   'bildstein_station5_xyz_intensity_rgb',
                   'domfountain_station1_xyz_intensity_rgb',
     