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
        T old_energy = energy_zero; //energy at the previous iteration
        //vector with time and energy, useful for benchmarking
        std::vector<T> energy_out(this->parameter.max_ite_main ),time_out(this->parameter.max_ite_main);
        TimeStack ts; ts.tic();
        //the main loop
        for (uint32_t ite_main = 1; ite_main <= this->parameter.max_ite_main; ite_main++)
        {
            //--------those two lines are the whole iteration-------------------------
            uint32_t saturation = this->split(); //compute optimal binary partition
            this->reduce(); //compute the new reduced graph
            //-------end of the iteration - rest is stopping check and display------
            std::pair<T,T> energy = this->compute_energy();
            energy_out.push_back((energy.first + energy.second));
            time_out.push_back(ts.tocDouble());
            if (this->parameter.verbose > 1)
            {
                printf("Iteration %3i - %4i components - ", ite_main, (int)this->components.size());
                printf("Saturation %5.1f %% - ",100*saturation / (double) this->nVertex);
                switch (this->parameter.fidelity)
                {
                    case L2:
                    {
                        printf("Quadratic Energy %4.3f %% - ", 100 * (energy.first + energy.second) / energy_zero);
                        break;
                    }
                    case linear:
                    {
                        printf("Linear Energy %10.1f - ", energy.first + energy.second);
                        break;
                    }
                    case KL:
                    {
                        printf("KL Energy %4.3f %% - ", 100 * (energy.first + energy.second) / energy_zero);
                        break;
                    }
       case SPG:
                    {
                        printf("Quadratic Energy %4.3f %% - ", 100 * (energy.first + energy.second) / energy_zero);
                        break;
                    }
                }
                std::cout << "Timer  " << ts.toc() << std::endl;
            }
            //----stopping checks-----
            if (saturation == (double) this->nVertex)
            {   //all components are saturated
                if (this->parameter.verbose > 1)
                {
                    std::cout << "All components are saturated" << std::endl;
                }
                break;
            }
            if ((old_energy - energy.first - energy.second) / (old_energy)
               < this->parameter.stopping_ratio)
            {   //relative energy progress stopping criterion
                if (this->parameter.verbose > 1)
                {
                    std::cout << "Stopping criterion reached" << std::endl;
                }
                break;
            }
            if (ite_main>=this->parameter.max_ite_main)
            {   //max number of iteration
                if (this->parameter.verbose > 1)
                {
                    std::cout << "Max number of iteration reached" << std::endl;
                }
                break;
            }
            old_energy = energy.first + energy.second;
        }
        if (this->parameter.cutoff > 0)
        {
            this->cutoff();
        }
        return std::pair<std::vector<T>, std::vector<T>>(energy_out, time_out);
    }
    //=============================================================================================
    //=========== VIRTUAL METHODS DEPENDING ON THE CHOICE OF FIDELITY FUNCTION =====================
    //=============================================================================================
    //
    //=============================================================================================
    //=============================        SPLIT        ===========================================
    //=============================================================================================
    virtual uint32_t split()
    {
        //compute the optimal binary partition
        return 0;
    }
    //=============================================================================================
    //================================     compute_energy_L2      ====================================
    //=============================================================================================
    virtual std::pair<T,T> compute_energy()
    {
        //compute the current energy
        return std::pair<T,T>(0,0);
    }
    //=============================================================================================
    //=================================   COMPUTE_VALUE   =========================================
    //=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_value(const uint32_t & ind_com)
    {
        //compute the optimal the values associated with the current partition
        return std::pair<std::vector<T>, T>(std::vector<T>(0),0);
    }
//=============================================================================================
//=================================   COMPUTE_MERGE_GAIN   =========================================
//=============================================================================================
    virtual std::pair<std::vector<T>, T> compute_merge_gain(const VertexDescriptor<T> & comp1
                                                          , const VertexDescriptor<T> & comp2)
    {
        //compute the gain of mergeing two connected components
        return std::pair<std::vector<T>, T>(std::vector<T>(0),0);
    }
    //=============================================================================================
    //========================== END OF VIRTUAL METHODS ===========================================
    //=============================================================================================
    //
    //=============================================================================================
    //=============================     INITIALIZE      ===========================================
    //=============================================================================================
    void initialize()
    {
        //build the reduced graph with one component, fill the first vector of components
        //and add the sink and source nodes
        VertexIterator<T> ite_ver, ite_ver_end;
        EdgeAttributeMap<T> edge_attribute_map
            = boost::get(boost::edge_bundle, this->main_graph);
        this->components[0]  = std::vector<VertexDescriptor<T>> (0);//(this->nVertex);
        this->root_vertex[0] = *boost::vertices(this->main_graph).first;
        this->nVertex = boost::num_vertices(this->main_graph);
        this->nEdge   = boost::num_edges(this->main_graph);
        //--------compute the first reduced graph----------------------------------------------------------
        for (boost::tie(ite_ver, ite_ver_end) = boost::vertices(this->main_graph);
             ite_ver != ite_ver_end; ++ite_ver)
        {
            this->components[0].push_back(*ite_ver);
        }
        this->lastIterator = ite_ver;
        this->compute_value(0);
        //--------build the link to source and sink--------------------------------------------------------
        this->source = boost::add_vertex(this->main_graph);
        this->sink   = boost::add_vertex(this->main_graph);
        uint32_t eIndex = boost::num_edges(this->main_graph);
        ite_ver = boost::vertices(this->main_graph).first;
        for (uint32_t ind_ver = 0;  ind_ver < this->nVertex ; ind_ver++)
        {
            // note that source and edge will have many nieghbors, and hence boost::edge should never be called to get
            // the in_edge. use the out_edge and then reverse_Edge
            addDoubledge<T>(this->main_graph, this->source, boost::vertex(ind_ver, this->main_graph),