import argparse

from RandLANet import Network, log_out
from sampler2 import *

if __name__ == '__main__':
    """create seed samples and model weights"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--dataset', type=str, default='semantic3d', choices=["S3DIS", "semantic3d", "SemanticKITTI"])
    parser.add_argument('--seed_percent', type=float, default=0.01, help='seed percent')
    parser.add_argument('--reg_strength', default=0.012, type=float,
                        help='regularization strength for the minimal partition')

    FLAGS = parser.parse_args()
    os.envir