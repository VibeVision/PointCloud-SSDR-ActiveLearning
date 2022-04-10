import argparse
import math
import time

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="2", help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    parser.add_argument('--sampler', type=str, default='T', choices=["random", "T"], help='sampler')
    parser.add_argument('--dataset', type=str, default='semantic3d', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--round', type=int, default=2)
    parser.add_argument('--classbal', type=int, default=0, choices=[0,1,2])
    parser.add_argument('--distance', type=int, default=0, choices=[0,1])

    parser.add_argument('--edcd', type=int, default=0, choices=[0,1])

    parser.add_argument('--uncertainty_mode', type=str, default="mean", choices=["mean", "sum_weight", "WetSU"], help='the mode from pixel uncertainty to region uncertainty')
    parser.add_argument('--point_uncertainty_mode', type=str, default="entropy", choices=["lc", "sb", "entropy"],
                        help='point uncertainty')

    parser.add_argument('--oracle_mode', type=str, default="dominant", choices=["dominant", "part_do", "NAIL", "domi_prec4", "domi_prec3"],
                        help='the mode from pixel uncertainty to region uncertainty. domi_prec4 denotes it begins using NAIL labeling when round 4 ')

    parser.add_argument('--reg_strength', default=0.008, type=float,
                        help='regularization strength for the minimal partition')
    parser.add_argument('--threshold', default=0.9, type=float,
                        help='tolerance threshold')
    parser.add_argument('--min_size', default=1, type=int,
                        help='the number of points in one selected superpoint >= min_size')

    parser.add_argument('--t', default=0, type=int,
                        help='t, multiple run')

    parser.add_argument('--gcn', default=0, type=int,
           