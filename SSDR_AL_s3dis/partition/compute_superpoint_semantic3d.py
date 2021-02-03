
import os.path
import argparse
from graphs import compute_graph_nn_2
# from provider import *
from helper_ply import read_ply
import glob
import pickle
import os
import numpy as np
import sys
sys.path.append("partition/cut-pursuit/build/src")
sys.path.append("cut-pursuit/build/src")
sys.path.append("ply_c")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c

# import pydevd_pycharm
# pydevd_pycharm.settrace('10.214.160.245', port=11111, stdoutToServer=True, stderrToServer=True)

train_cloud_name_list = ['bildstein_station1_xyz_intensity_rgb',
                             'bildstein_station5_xyz_intensity_rgb',
                             'domfountain_station1_xyz_intensity_rgb',
                             'domfountain_station2_xyz_intensity_rgb',
                             'domfountain_station3_xyz_intensity_rgb',
                             'neugasse_station1_xyz_intensity_rgb',
                             'sg27_station1_intensity_rgb',
                             'sg27_station4_intensity_rgb',
                             'sg27_station5_intensity_rgb',
                             'sg27_station9_intensity_rgb',
                             'sg28_station4_intensity_rgb',
                             'untermaederbrunnen_station1_xyz_intensity_rgb',
                             'untermaederbrunnen_station3_xyz_intensity_rgb']

val_cloud_name_list = ['bildstein_station3_xyz_intensity_rgb',
                       'sg27_station2_intensity_rgb']

test_cloud_name_list = ['MarketplaceFeldkirch_Station4_rgb_intensity-reduced',
                        'sg27_station10_rgb_intensity-reduced',
                        'sg28_Station2_rgb_intensity-reduced',
                        'StGallenCathedral_station6_rgb_intensity-reduced']

def semantic3d_superpoint(args):
    path = "data/semantic3d"

    output_dir = os.path.join(path, str(args.reg_strength), "superpoint")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tree_path = os.path.join(path, 'input_{:.3f}'.format(0.06))

    total_obj = {}
    total_obj["unlabeled"] = {}
    sp_num = 0
    file_num = 0
    point_num = 0

    for cloud_name in train_cloud_name_list:
        sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
        data = read_ply(sub_ply_file)
        rgb = np.vstack((data['red'], data['green'], data['blue'])).T
        xyz = np.vstack((data['x'], data['y'], data['z'])).T
        # xyz = xyz.astype('f4')
        # rgb = rgb.astype('uint8')
        # ---compute 10 nn g