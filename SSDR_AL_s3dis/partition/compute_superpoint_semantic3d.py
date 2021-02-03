
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

train_cloud_name_list = ['bildstein_statio