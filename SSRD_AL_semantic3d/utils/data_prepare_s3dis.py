from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import pandas as pd
import os, sys, glob, pickle

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool import DataProcessing as DP

dataset_path = './data/S3DIS/Stanf