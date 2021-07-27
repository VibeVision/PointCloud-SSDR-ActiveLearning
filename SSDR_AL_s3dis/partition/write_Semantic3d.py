#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import glob
import numpy as np
import argparse
from provider import *
parser = argparse.ArgumentParser(description='123434234')
parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3D')
parser.add_argument('--odir', default='./results/semantic3d', help='Directory to store results')
parser.add_argument('--ver_batch', default=5000000, type=int, help='Batch size for reading large files')
parser.add_argument('--db_test_name', default='testred')
args = parser.parse_args()
#---path to data---------------------------------------------------------------
#root of the data directory
root = args.SEMA3D_PATH+'/'
#list of subfolders to be processed
if args.db_test_name == 'testred':
    area = 'test_reduced/'
elif args.db_test_name == 'testfull':
    area = 'test_full/'
#------------------------------------------------------------------------------
print("=================\n   " + area + "\n=================")
data_folder = root + "data/"               + area
fea_folder  = root + "features/"           + area
spg_folder  = root + "superpoint_graphs/"           + area
res_folder  = './' + args.odir + '/'
labels_folder =  root + "labels/"          + area
if not os.path.isdir(data_folder):
    raise ValueError(