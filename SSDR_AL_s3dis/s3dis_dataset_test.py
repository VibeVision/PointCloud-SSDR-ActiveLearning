import numpy as np
import glob
import pickle
import time
from os.path import join

import numpy as np

from helper_ply import read_ply
from helper_tool import ConfigS3DIS
from helper_tool import DataProcessing as DP


class S3DIS_Dataset_Test:
    def __init__(self, test_area_idx):
        self.name = 'S3DIS'
        self.path = './data/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'input_{:.3f}'.format(ConfigS3DIS.sub_grid_size), '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_names = []

        self.load_sub_sampled_clouds(ConfigS3DIS.sub_grid_size)
        self.init_possibility()

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                # Name of the input files
                kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
                sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

                data = read_ply(sub_ply_file)
                sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T   # shape=[point_number, 3]
                sub_labels = data['class']  # shape=[point_number]

                # Read pkl with search tree
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)


                self.input_trees += [search_tree]
                self.input_colors += [sub_colors]
                self.input_labels += [sub_labels]
  