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
                               7: 'tab