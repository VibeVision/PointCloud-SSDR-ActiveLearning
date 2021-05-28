#include <iostream>
#include <vector>
#include "mex.h"
//#include <opencv2/opencv.hpp>
#include "../include/API.h"
//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//Greedy graph cut based algorithm to solve the generalzed minimal 
//partition problem
//
//Cut Pursuit: fast algorithms to learn piecewise constant functions on 
//general weighted graphs, Loic Landrieu and Guillaume Obozinski,2016.
//
//Produce a piecewise constant approximation of signal $y$ structured
//by the graph G=(V,e,mu,w) with mu the node weight and w the edge_weight:
//argmin \sum_{i \IN V}{mu_i * phi(x_I, y_I)} 
//+ \sum_{(i,j) \IN E}{w_{i,j} 1(x_I != x_J)}
//
//phi(X,Y) the fidelity function (3 are implemented)
//(x != y) the funciton equal to 1 if x!=y and 0 else
//
// LOIC LANDRIEU 2017
//
//=======================SYNTAX===================================================
//
//[solution, inComponent, components, Eu_red, Ev_red, edge_weight_red, node_weight_red, vertices_border]
// = L0_cut_pursuit_segmentation(observation, Eu, Ev, lambda = 1, edge_weight = [1 ... 1]
//                 , node_weight = [1 ... 1], mode = 1, speed = 1, verbose = false)
//-----INPUT-----
// N x D float observation : the observed signal
// E x 1 int Eu, Ev: the origin and destination of each node
// E x 1 float  edge_weight: the edge weight
// N x 1 float  node_weight: the node weight
// 1 x 1 float lambda : the regularization strength
// 1 x 1 float mode : the fidelity function
//      0 : linear (for simplex bound data)
//      1 : quadratic (default)
//   0<a<1: KL with a smoothing (for simplex bound data)
// 1 x 1 float speed : parametrization impacting performance
//      0 : slow but precise
//      1 : recommended (default)
//      2 : fast but approximated (no backward step)
//      3 : ludicrous - for prototyping (no backward step)
// 1 x 1 b