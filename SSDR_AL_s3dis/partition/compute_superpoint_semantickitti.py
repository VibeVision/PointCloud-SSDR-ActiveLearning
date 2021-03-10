
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

def semantickitti_superpoint(args, val_split):
    path = "data/SemanticKITTI"
    all_files = glob.glob(os.path.join(path, 'input_{:.3f}'.format(0.06), '*.ply'))

    output_dir = os.path.join(path, str(args.reg_strength), "superpoint")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tree_path = os.path.join(path, 'input_{:.3f}'.format(0.06))

    total_obj = {}
    total_obj["unlabeled"] = {}
    sp_num = 0
    file_num = 0
    point_num = 0

    for i, file_path in enumerate(all_files):
        print(file_path)
        cloud_name = file_path.split('/')[-1][:-4]  # 获取去掉后缀的文件名
        if val_split not in cloud_name:
            sub_ply_file = os.path.join(tree_path, '{:s}.ply'.format(cloud_name))
            data = read_ply(sub_ply_file)
            xyz = np.vstack((data['x'], data['y'], data['z'])).T  # shape=[point_number, 3]
            # xyz = xyz.astype('f4')
            # rgb = rgb.astype('uint8')
            # ---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)
            # ---compute geometric features-------
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype(
                'float32')  # shape=【point_number，4】 。每个point聚合k_nn_geof个临近points,生成包含【linearity，planarity，scattering，verticality】的特征
            del target_fea
            # --compute the partition------
            # --- build the spg h5 file --
            features = geof
            geof[:, 3] = 2. * geof[:, 3]

            graph_nn["edge_weight"] = np.array(
                1. / (args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])),
                dtype='float32')
            print("minimal partition...")

            # components [sp_idx, point_ids]  是一个二维list ,
            # in_component  [point_idx, sp_idx], in_component 中的in 就是 jndex 缩写
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"]
                                                      