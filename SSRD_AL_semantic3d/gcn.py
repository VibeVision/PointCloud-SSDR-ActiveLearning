import time

import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F
import torch.optim as optim
from kcenterGreedy import *
from os.path import join
import pickle
from helper_ply import read_ply
from sklearn.neighbors import KDTree

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True, gcn_gpu=1):
        super(GraphConvolution, self).__init__()
        self.gcn_gpu = gcn_gpu
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).cuda(device=self.gcn_gpu))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).cuda(device=self.gcn_gpu))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
         A * (V * W) + b:   [N, N] * ([N, m] * [m, hidden]) + [hidden]  => [N, hidden]
        Ar