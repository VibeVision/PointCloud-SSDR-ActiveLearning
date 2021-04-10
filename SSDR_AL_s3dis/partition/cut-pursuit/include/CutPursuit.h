#pragma once
#include "Graph.h"
#include <math.h>
#include <queue>
#include <iostream>
#include <fstream>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
namespace CP {

template <typename T>
struct CPparameter
{
    T   reg_strenth;  //regularization strength, multiply the edge weight
    uint32_t cutoff;  //minimal component size
    uint32_t flow_steps; //number of steps in the optimal binary cut computation
    uint32_t kmeans_ite; //number of iteration in the kmeans sampling
    uint32_t kmeans_resampling; //number of kmeans re-intilialization
    uint32_t verbose; //verbosity
    uint32_t max_ite_main; //max number of iterations in the main loop
    bool backward_step; //indicates if a backward step should be performed
    double stopping_ratio; //when (E(t-1) - E(t) / (E(0) - E(t)) is too small, the algorithm stops
    fidelityType fidelity; //the fidelity function
    double smoothing; //smoothing term (for Kl divergence only)
    bool parallel; //enable/disable parrallelism
	T weight_decay; //for continued optimization of the flow steps
};

template <typename T>
class CutPursuit
{
    public:
    Graph<T> main_graph; //the Graph structure containing the main structure
    Graph<T> reduced_graph; //the reduced graph whose vertices are the connected component
    std::vector<std::vector<VertexDescriptor<T>>> components; //contains the list of the vertices in each component
    std::vector<VertexDescriptor<T>> root_vertex; //the root vertex for each connected components
    std::vector<bool> saturated_components; //is the component saturated (uncuttable)
    std::vector<std::vector<EdgeDescriptor>> borders; //the list of edges forming the borders between the connected components
    VertexDescriptor<T> source; //source vertex for graph cut
    VertexDescriptor<T> sink; //sink vertex
    ui