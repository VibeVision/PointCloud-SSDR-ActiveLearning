
import os
import sys
import random
import glob
import numpy as np
#from numpy import genfromtxt
import pandas as pd
import h5py
#import laspy
from sklearn.neighbors import NearestNeighbors


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
# sys.path.insert(0, os.path.join(DIR_PATH, '..'))
# from partition.ply_c import libply_c
# import colorsys
sys.path.append("partition/cut-pursuit/build/src")
sys.path.append("cut-pursuit/build/src")
sys.path.append("ply_c")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libply_c
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement

#------------------------------------------------------------------------------
def partition2ply(filename, xyz, components):
    """write a ply with random colors for each components"""
    random_color = lambda: random.randint(0, 255)
    color = np.zeros(xyz.shape)
    for i_com in range(0, len(components)):
 