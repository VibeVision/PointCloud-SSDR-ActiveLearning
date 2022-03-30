#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os.path
import glob
import numpy as np
import argparse
from provider import *
parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--SEMA3D_PATH', default='datasets/semantic3D')
parser.add_argument('--odir', default='./results/semantic3d', help='Directory to store results')
parser.add_argument('--ver_batch', default=5000000, type=int, help=