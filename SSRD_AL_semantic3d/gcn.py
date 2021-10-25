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
        Args:
            input:  V
            adj:    A

        Returns:

        """
        support = torch.mm(input, self.weight).cuda(device=self.gcn_gpu)
        output = torch.spmm(adj, support).cuda(device=self.gcn_gpu)
        if self.bias is not None:
            return (output + self.bias).cuda(device=self.gcn_gpu)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    """
    A * ((A * (V * W1) + b1) * W2) + b2
    """

    def __init__(self, nfeat, nhid, nclass, dropout, gcn_gpu):
        super(GCN, self).__init__()
        self.gcn_gpu = gcn_gpu
        self.gc1 = GraphConvolution(nfeat, nhid).cuda(device=gcn_gpu)
        self.gc2 = GraphConvolution(nhid, nhid).cuda(device=gcn_gpu)
        self.gc3 = GraphConvolution(nhid, nclass).cuda(device=gcn_gpu)
        self.dropout = dropout
        self.linear = nn.Linear(nclass, 1).cuda(device=gcn_gpu)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj)).cuda(device=self.gcn_gpu)
        feat = F.dropout(x, self.dropout, training=self.training).cuda(device=self.gcn_gpu)
        x = self.gc3(feat, adj)

        return torch.sigmoid(x), feat, torch.cat((feat,x),1).cuda(device=self.gcn_gpu)

def BCEAdjLoss(scores, lbl, nlbl, l_adj, gcn_gpu):
    lnl = torch.log(scores[lbl]).cuda(device=gcn_gpu)
    lnu = torch.log(1 - scores[nlbl]).cuda(device=gcn_gpu)
    labeled_score = torch.mean(lnl).cuda(device=gcn_gpu)
    unlabeled_score = torch.mean(lnu).cuda(device=gcn_gpu)
    bce_adj_loss = (-labeled_score - l_adj*unlabeled_score).cuda(device=gcn_gpu)
    return bce_adj_loss

def chamfer_distance(cloud_list, tree_list, centroid_idx):
    """numpy"""
    centroid_cloud = cloud_list[centroid_idx]
    centroid_tree = tree_list[centroid_idx]
    distances = np.zeros([len(cloud_list)])
    for i in range(len(cloud_list)):
        if i != centroid_idx:
            distances1, _ = centroid_tree.query(cloud_list[i])
            distances2, _ = tree_list[i].query(centroid_cloud)
            av_dist1 = np.mean(distances1)
            av_dist2 = np.mean(distances2)
            distances[i] = av_dist1 + av_dist2
    return distances

def create_cd(superpoint_list, superpoint_centroid_list):
    """numpy"""
    sp_num = len(superpoint_list)
    cd_dist = np.zeros([sp_num, sp_num])
    align_superpoint_list = []
    tree_list = []
    for i in range(sp_num):
        # 中心对齐
        align_superpoint = superpoint_list[i] - superpoint_centroid_list[i]
        align_superpoint_list.append(align_superpoint)
        tree_list.append(KDTree(align_superpoint))
    for i in range(sp_num):
        cd_dist[i] = chamfer_distance(align_superpoint_list, tree_list, i)
    return cd_dist

def create_adj(featuresV, labeled_select_ref, unlabeled_candidate_ref, input_path, data_path, gcn_gpu):
    begin_time = time.time()
    print("1")
    unlabeled_num = len(unlabeled_candidate_ref)
    print("2")
    labeled_num = len(labeled_select_ref)
    print("3")
    N = unlabeled_num + labeled_num
    total_cloud = {}  # {cloud_name: [{sp_idx, ref_idx}]}
    cloud_name_list = []
    for i in range(unlabeled_num):
        print("3", i)
        cloud_name = unlabeled_candidate_ref[i]["cloud_name"]
        sp_idx = unlabeled_candidate_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": i})
    print("4")
    for i in range(labeled_num):
        print("4", i)
        cloud_name = labeled_select_ref[i]["cloud_name"]
        sp_idx = labeled_select_ref[i]["sp_idx"]
        if cloud_name not in total_cloud:
            total_cloud[cloud_name] = []
            cloud_name_list.append(cloud_name)
        total_cloud[cloud_name].append({"sp_idx": sp_idx, "ref_idx": unlabeled_num+i})

    print("ed,cd below")
    A_ed = np.ones([N, N], dtype=np.