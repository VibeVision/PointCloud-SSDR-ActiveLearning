from open3d import linux as open3d
from os.path import join
import numpy as np
import colorsys, random, os, sys
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


class ConfigSemanticKITTI:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 4096 * 11  # Number of input points
    num_classes = 19  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 6  # batch_size during training
    val_batch_size = 20  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]

    noise_init = 3.5  # noise initial parameter
    max_epoch = 100  # maximum epoch during training
    learning_rate = 1e-2  # initial learning rate
    lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    # add by ssdr
    ignored_label_inds = [0]

class ConfigSemantic3D:
    k_n = 16  # KNN
    num_layers = 5  # Number of layers
    num_points = 65536  # Number of input points
    num_classes = 8  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    # batch_size = 4  # batch_size during training
    batch_size = 3  # batch_size during training
    val_batch_size = 16  # batch_size during validation and test
    # train_steps = 500  # Number of steps per epochs
    train_steps = 667  # Number of steps per epochs
    # val_steps = 100  # Number of validation steps per epoch 下午刚改的
    val_steps = 50  # Number of validation steps per epoch
    ### add ssdr
    # test_batch_size = 16  # batch_size during validation and test
    # test_steps = 100  # Number of validation steps per epoch
    ###

    sub_sampling_ratio = [4, 4, 4, 4, 2]  # sampling ratio of random sampling at each layer
    d_out = [16, 64, 128, 256, 512]  # feature dimension

    noise_init = 3.5  # noise initial parameter
    # max_epoch = 100  # maximum epoch during training
    # max_epoch = 50  # maximum epoch during training   0.84 ** 30 == 0.95 ** 100
    max_epoch = 30  # maximum epoch during training   0.84 ** 30 == 0.95 ** 100
    learning_rate = 1e-2  # initial learning rate
    # lr_decays = {i: 0.95 for i in range(0, 500)}  # decay rate of learning rate
    # lr_decays = {i: 0.90 for i in range(0, 500)}  # decay rate of learning rate
    lr_decays = {i: 0.92 for i in range(0, 500)}  # decay rate of learning rate

    train_sum_dir = 'train_log'
    saving = True
    saving_path = None

    augment_scale_anisotropic = True
    augment_symmetries = [True, False, False]
    augment_rotation = 'vertical'
    augment_scale_min = 0.8
    augment_scale_max = 1.2
    augment_noise = 0.001
    augment_occlusion = 'none'
    augment_color = 0.8
    # add by ssdr
    ignored_label_inds = []
    # ignored_label_inds = [0]

class DataProcessing:
    @staticmethod
    def load_pc_semantic3d(filename):
        pc_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.float16)
        pc = pc_pd.values
        return pc

    @staticmethod
    def load_label_semantic3d(filename):
        label_pd = pd.read_csv(filename, header=None, delim_whitespace=True, dtype=np.uint8)
  