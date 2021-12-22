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
// 1 x 1 bool verose : verbosity
//      0 : silent
//      1 : recommended (default)
//      2 : chatty
//-----OUTPUT-----
// N x 1 float  solution: piecewise constant approximation
// N x 1 int inComponent: for each node, in which component it belongs
// n_nodes_red x 1 cell components : for each component, list of the nodes
// 1 x 1 int n_nodes_red : number of components
// 1 x 1 int n_edges_red : number of edges in reduced graph
// n_edges_red x 1 int Eu_red, Ev_red : source and target of reduced edges
// n_edges_red x 1 float edge_weight_red: weights of reduced edges
// n_nodes_red x 1  float node_weight_red: weights of reduced nodes
// n_edges_red x 1 cell vertices_border: for each edge of the reduced graph,
//  the list of index of the edges composing the interface between components


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //---read dimensions----//
    const int n_nodes = mxGetN(prhs[0]); //number of nodes
    const int nObs = mxGetM(prhs[0]); // size of observation for each node
    const int nEdg = mxGetNumberOfElements(prhs[1]); //number of edges
    //---read inputs----
    std::vector< std::vector<float> > observation(n_nodes);
    const std::vector<uint32_t> Eu((uint32_t*)mxGetData(prhs[1]), (uint32_t*)mxGetData(prhs[1])+nEdg);
    const std::vector<uint32_t> Ev((uint32_t*)mxGetData(prhs[2]), (uint32_t*)mxGetData(prhs[2])+nEdg);
    const float lambda        = (float) mxGetScalar(prhs[3]); //reg strength
    const std::vector<float> edge_weight((float*)mxGetData(prhs[4]), (float*)mxGetData(prhs[4])+nEdg);
    const std::vector<float> node_weight((float*)mxGetData(prhs[5]), (float*)mxGetData(prhs[5])+n_nodes);
    const float mode          = (float) mxGetScalar(prhs[6]); //fidelity
    const uint32_t cutoff     = (uint32_t) mxGetScalar(prhs[7]); //cut off
    const float speed         = (float) mxGetScalar(prhs[8]); //speed mode
    const float weight_decay  = (float) mxGetScalar(prhs[9]); //weight decay*/
    const float verbose       = (float) mxGetScalar(prhs[10]); //verbosity*/
    //--fill the observation----
    const float *observation_array = (float*) mxGetData(prhs[0]);
    std::vector< std::vector<float> > solution(n_nodes);
    int ind_obs = 0;
    for (int ind_nod = 0; ind_nod < n_nodes; ind_nod++)
    {
        observation[ind_nod] = std::vector<float>(nObs);
        solution[ind_nod]    = std::vector<float>(nObs);
        for (int ind_dim = 0; ind_dim < nObs; ind_dim++)
        {   
            observation[ind_nod][ind_dim] = observation_array[ind_obs];
            ind_obs++;
        }
    }
    //---set up output--------------------------------
    //plhs[0] = mxDuplicateArray(prhs[0]);
    uint32_t * in_component_array, * Eu_red_array, * Ev_red_array;
    float * edge_weight_red_array, * node_weight_red_array;
    