#pragma once
#include <omp.h>
#include "CutPursuit_L2.h"
#include "CutPursuit_Linear.h"
#include "CutPursuit_KL.h"
#include "CutPursuit_SPG.h"
//**********************************************************************************
//*******************************L0-CUT PURSUIT*************************************
//**********************************************************************************
//Greedy graph cut based algorithm to solve the generalized minimal
//partition problem
//
//Cut Pursuit: fast algorithms to learn piecewise constant functions on
//general weighted graphs, Loic Landrieu and Guillaume Obozinski,2016.
//
//Produce a piecewise constant approximation of signal $y$ structured
//by the graph G=(V,e,mu,w) with mu the node weight and w the edgeweight:
//argmin \sum_{i \IN V}{mu_i * phi(x_I, y_I)}
//+ \sum_{(i,j) \IN E}{w_{i,j} 1(x_I != x_J)}
//
//phi(X,Y) the fidelity function (3 are implemented)
//(x != y) the function equal to 1 if x!=y and 0 else
//
// LOIC LANDRIEU 2017
//
//=======================SYNTAX===================================================
//---------------REGULARIZATION---------------------------------------------------
//C style inputs
//void cut_pursuit(const uint32_t n_nodes, const uint32_t n_edges, const uint32_t nObs
//          ,const T * observation, const uint32_t * Eu, const uint32_t * Ev
//          ,const T * edgeWeight, const T * nodeWeight
//          ,T * solution,  const T lambda, const uint32_t cutoff, const T mode, const T speed, const T weight_decay
//          , const float verbose)
//C++ style input
//void cut_pursuit(const uint32_t n_nodes, const uint32_t n_edges, const uint32_t nObs
//          , std::vector< std::vector<T> > & observation
//          , const std::vector<uint32_t> & Eu, const std::vector<uint32_t> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector< std::vector<T> > & solution,  const T lambda, const uint32_t cutoff, const T mode, const T speed, const T weight_decay
//          , const float verbose)
// when D = 1
//void cut_pursuit(const uint32_t n_nodes, const uint32_t n_edges, const uint32_t nObs
//          , std::vector<T> & observation
//          , const std::vector<uint32_t> & Eu, const std::vector<uint32_t> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector<T> & solution,  const T lambda, const uint32_t cutoff, const T mode, const T speed
//         , const float verbose)

//-----INPUT-----
// 1x1 uint32_t n_nodes = number of nodes
// 1x1 uint32_t n_edges = number of edges
// 1x1 uint32_t nObs   = dimension of data on each node
// NxD float observation : the observed signal
// Ex1 uint32_t Eu, Ev: the origin and destination of each node
// Ex1 float edgeWeight: the edge weight
// Nx1 float nodeWeight: the node weight
// 1x1 float lambda : the regularization strength
// 1x1 uint32_t cutoff : minimal component size
// 1x1 float mode : the fidelity function
//      0 : linear (for simplex bound data)
//      1 : quadratic (default)
//   0<a<1: KL with a smoothing (for simplex bound data)
// 1x1 float speed : parametrization impacting performance
//      0 : slow but precise
//      1 : recommended (default)
//      2 : fast but approximated (no backward step)
//      3 : ludicrous - for prototyping (no backward step)
// 1x1 float weight_decay : weight decay to compute the optimal binary partition
// 1x1 bool verose : verbosity
//      0 : silent
//      1 : recommended (default)
//      2 : chatty
//-----OUTPUT-----
// Nx1 float  solution: piecewise constant approximation
// Nx1 uint32_t inComponent: for each node, in which component it belongs
// n_node_redx1 cell components : for each component, list of the nodes
// 1x1 n_node_red : number of components
// 1x1 uint32_t n_edges_red : number of edges in reduced graph
// n_edges_redx1 uint32_t Eu_red, Ev_red : source and target of reduced edges
// n_edges_redx1 float edgeWeight_red: weights of reduced edges
// n_node_redx1  float nodeWeight_red: weights of reduced nodes
//---------------SEGMENTATION--------------------------------------------------
//for the segmentation, the functions has a few extra argumens allowing to
//record the structrue of the reduced graph
//C++ style input
//void cut_pursuit(const uint32_t n_nodes, const uint32_t n_edges, const uint32_t nObs
//          , std::vector< std::vector<T> > & observation
//          , const std::vector<uint32_t> & Eu, const std::vector<uint32_t> & Ev
//          ,const std::vector<T> & edgeWeight, const std::vector<T> & nodeWeight
//          ,std::vector< std::vector<T> > & solution,
//          , const std::vector<uint32_t> & in_component
//	    , std::vector< std::vector<uint32_t> > & components
//          , uint32_t & n_nodes_red, uint32_t & n_edges_red
//          , std::vector<uint32_t> & Eu_red, std::vector<uint32_t> & Ev_red
//          , std::vector<T> & edgeWeight_red, std::vector<T> & nodeWeight_red
//  	    , const T lambda, const T mode, const T speed, const T weight_decay
//          , const float verbose)
//-----EXTRA INPUT-----
// Nx1 uint32_t inComponent: for each node, in which component it belongs
// 1x1 n_node_red : number of components
// 1x1 uint32_t n_edges_red : number of edges in reduced graph
// n_node_redx1 cell component