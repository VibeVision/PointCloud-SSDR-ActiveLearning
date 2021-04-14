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
    uint32_t dim;     // dimension of the data
    uint32_t nVertex; // number of data point
    uint32_t nEdge;   // number of edges between vertices (not counting the edge to source/sink)
    CP::VertexIterator<T> lastIterator; //iterator pointing to the last vertex which is neither sink nor source
    CPparameter<T> parameter;
    public:      
    CutPursuit(uint32_t nbVertex = 1)
    {
        this->main_graph     = Graph<T>(nbVertex);
        this->reduced_graph  = Graph<T>(1);
        this->components     = std::vector<std::vector<VertexDescriptor<T>>>(1);
        this->root_vertex    = std::vector<VertexDescriptor<T>>(1,0);
        this->saturated_components = std::vector<bool>(1,false);
        this->source         = VertexDescriptor<T>();
        this->sink           = VertexDescriptor<T>();
        this->dim            = 1;
        this->nVertex        = 1;
        this->nEdge          = 0;
        this->parameter.flow_steps  = 3;
        this->parameter.kmeans_ite  = 5;
        this->parameter.kmeans_resampling = 3;
        this->parameter.verbose = 2;
        this->parameter.max_ite_main = 6;
        this->parameter.backward_step = true;
        this->parameter.stopping_ratio = 0.0001;
        this->parameter.fidelity = L2;
        this->parameter.smoothing = 0.1;
        this->parameter.parallel = true;
		this->parameter.weight_decay = 0.7;
    }
    virtual ~CutPursuit(){
    };  
    //=============================================================================================
    std::pair<std::vector<T>, std::vector<T>> run()
    {
        //first initilialize the structure
        this->initialize();
        if (this->parameter.verbose > 0)
        {
            std::cout << "Graph "  << boost::num_vertices(this->main_graph) << " vertices and "
             <<   boost::num_edges(this->main_graph)  << " edges and observation of dimension "
             << this->dim << '\n';
        }
        T energy_zero = this->compute_energy().first; //energy with 1 component
        T old_energy = e