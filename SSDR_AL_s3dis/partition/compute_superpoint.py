
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

def s3dis_superpoint(args, val_split):
    path = "data/S3DIS"
    all_files = glob.glob(os.path.join(path, 'original_ply', '*.ply'))

    output_dir = os.path.join(path, str(args.reg_strength), "superpoint")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tree_path = os.path.join(path, 'input_{:.3f}'.format(0.04))

    total_obj = {}
    total_obj["unlabeled"] = {}
    sp_num = 0
    file_num = 0
    point_num = 0

    for i, file_path in enumerate(all_files):
        print(file_path)
        cloud_name = file_path.split('/')[-1][:-4]
        if val_split not in cloud_name:
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)
            rgb = np.vstack((data['red'], data['green'], data['blue'])).T
            xyz = np.vstack((data['x'], data['y'], data['z'])).T
            # xyz = xyz.astype('f4')
            # rgb = rgb.astype('uint8')
            # ---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            # ---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype(
                'float32')
            del target_fea
            # --compute the partition------
            # --- build the spg h5 file --
            features = np.hstack((geof, rgb)).astype('float32')  # add rgb as a feature for partitioning
   