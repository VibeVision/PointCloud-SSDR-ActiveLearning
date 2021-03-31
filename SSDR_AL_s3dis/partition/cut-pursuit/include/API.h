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
// n_node_redx1 cell components : for each component, list of the nodes
// n_edges_redx1 uint32_t Eu_red, Ev_red : source and target of reduced edges
// n_edges_redx1 float edgeWeight_red: weights of reduced edges
// n_node_redx1  float nodeWeight_red: weights of reduced nodes


namespace CP {
//===========================================================================
//=====================    CREATE_CP      ===================================
//===========================================================================

template<typename T>
CP::CutPursuit<T> * create_CP(const T mode, const float verbose)
{
    CP::CutPursuit<float> * cp = NULL;
    fidelityType fidelity = L2;
    if (mode == 0)
    {
        if (verbose > 0)
        {
            std::cout << " WITH LINEAR FIDELITY" << std::endl;
        }
        fidelity = linear;
        cp = new CP::CutPursuit_Linear<float>();
     }
     else if (mode == 1)
     {
        if (verbose > 0)
        {
            std::cout << " WITH L2 FIDELITY" << std::endl;
        }
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     else if (mode > 0 && mode < 1)
     {
        if (verbose > 0)
        {
            std::cout << " WITH KULLBACK-LEIBLER FIDELITY SMOOTHING : "
                      << mode << std::endl;
        }
        fidelity = KL;
        cp = new CP::CutPursuit_KL<float>();
        cp->parameter.smoothing = mode;
     }
	 else if (mode == -1)
	 {
        if (verbose > 0)
        {
            std::cout << " WITH ALTERNATE L2 NORM : "
                      << mode << std::endl;
        }
        fidelity = SPG;
        cp = new CP::CutPursuit_KL<float>();
        cp->parameter.smoothing = mode;
	 }
	 else if (mode == 2)
	 {
        if (verbose > 0)
        {
            std::cout << " PARTITION MODE WITH SPATIAL INFORMATION : "
                      << mode << std::endl;
        }
        fidelity = SPG;
        cp = new CP::CutPursuit_SPG<float>();
        cp->parameter.smoothing = mode;
	 }		
     else
     {
        std::cout << " UNKNOWN MODE, SWICTHING TO L2 FIDELITY"
                << std::endl;
        fidelity = L2;
        cp = new CP::CutPursuit_L2<float>();
     }
     cp->parameter.fidelity = fidelity;
     cp->parameter.verbose  = verbose;
     return cp;
}
//===========================================================================
//=== cut_pursuit segmentation C-style - with reduced graph  ================
//===========================================================================
template<typename T>
void cut_pursuit(const uint32_t n_nodes, const uint32_t n_edges, const uint32_t nObs
          , const T * observation
          , const uint32_t * Eu, const uint32_t * Ev
          , const T * edgeWeight, const T * nodeWeight
          , T * solution
	      , std::vector< uint32_t > & in_component, std::vector< std::vector<uint32_t> > & components
          , uint32_t & n_nodes_red, uint32_t & n_edges_red
          , std::vector< uint32_t > & Eu_red, std::vector<uint32_t> & Ev_red
          , std::vector< T > & edgeWeight_red, std::vector< T > & nodeWeight_red
  	      , const T lambda, const uint32_t cutoff, const T mode, const T speed, const T weight_decay
          , const float verbose)
{   //C-style ++ interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);

    set_speed(cp, speed, weight_decay, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
	cp->parameter.cutoff = cutoff;
    //-------run the optimization------------------------------------------
    cp->run();
    cp->compute_reduced_graph();
    //------------resize the vectors-----------------------------
    n_nodes_red = boost::num_vertices(cp->reduced_graph);
    n_edges_red = boost::num_edges(cp->reduced_graph);
    in_component.resize(n_nodes);
    components.resize(n_nodes_red);
    Eu_red.resize(n_edges_red);
    Ev_red.resize(n_edges_red);
    edgeWeight_red.resize(n_edges_red);
    nodeWeight_red.resize(n_nodes_red);
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    uint32_t ind_sol = 0;	
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(uint32_t ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
        for(uint32_t i_dim=0; i_dim < nObs; i_dim++)
        {
            solution[ind_sol] = vertex_attribute_map[*ite_nod].value[i_dim];
            ind_sol++;
        }
        ite_nod++;
   }
   //------------fill the components-----------------------------
    VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, cp->main_graph);
    for(uint32_t ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	uint32_t component_size = cp->components[ind_nod_red].size();
        components[ind_nod_red] = std::vector<uint32_t>(component_size, 0);
		for(uint32_t ind_nod = 0; ind_nod < component_size; ind_nod++ )
		{
			components[ind_nod_red][ind_nod] = vertex_index_map(cp->components[ind_nod_red][ind_nod]);
		}	
    }
    //------------write the reduced graph-----------------------------
    VertexAttributeMap<T> vertex_attribute_map_red = boost::get(
            boost::vertex_bundle, cp->reduced_graph);
    EdgeAttributeMap<T> edge_attribute_map_red = boost::get(
            boost::edge_bundle, cp->reduced_graph);
    VertexIndexMap<T> vertex_index_map_red = get(boost::vertex_index, cp->reduced_graph);
    ite_nod = boost::vertices(cp->main_graph).first;
    for(uint32_t ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {
        in_component[ind_nod] = vertex_attribute_map[*ite_nod].in_component;
        ite_nod++;
    }
    ite_nod = boost::vertices(cp->reduced_graph).first;
    for(uint32_t ind_nod_red = 0; ind_nod_red < n_nodes_red; ind_nod_red++ )
    {
	nodeWeight_red[ind_nod_red] = vertex_attribute_map_red[*ite_nod].weight;
        ite_nod++;
    }
    EdgeIterator<T> ite_edg = boost::edges(cp->reduced_graph).first;
    for(uint32_t ind_edg = 0; ind_edg < n_edges_red; ind_edg++ )
    {    
	edgeWeight_red[ind_edg] = edge_attribute_map_red[*ite_edg].weight;
	Eu_red[ind_edg] = vertex_index_map_red(boost::source(*ite_edg, cp->reduced_graph));
	Ev_red[ind_edg] = vertex_index_map_red(boost::target(*ite_edg, cp->reduced_graph));        
        ite_edg++;
    }
    delete cp;
    return;
}



//===========================================================================
//=====================  cut_pursuit  C-style  ==============================
//===========================================================================
template<typename T>
void cut_pursuit(const uint32_t n_nodes, const uint32_t n_edges, const uint32_t nObs
          ,const T * observation, const uint32_t * Eu, const uint32_t * Ev
          ,const T * edgeWeight, const T * nodeWeight
          ,T * solution, const T lambda, const uint32_t cutoff, const T mode, const T speed, const T weight_decay
          , const float verbose)
{   //C-style interface
    std::srand (1);
    if (verbose > 0)
    {
        std::cout << "L0-CUT PURSUIT";
    }
    //--------parameterization---------------------------------------------
    CP::CutPursuit<T> * cp = create_CP(mode, verbose);
    set_speed(cp, speed, weight_decay, verbose);
    set_up_CP(cp, n_nodes, n_edges, nObs, observation, Eu, Ev
             ,edgeWeight, nodeWeight);
    cp->parameter.reg_strenth = lambda;
    cp->parameter.cutoff = cutoff;
    //-------run the optimization------------------------------------------
    cp->run();
    //------------write the solution-----------------------------
    VertexAttributeMap<T> vertex_attribute_map = boost::get(
            boost::vertex_bundle, cp->main_graph);
    uint32_t ind_sol = 0;	
    VertexIterator<T> ite_nod = boost::vertices(cp->main_graph).first;
    for(uint32_t ind_nod = 0; ind_nod < n_nodes; ind_nod++ )
    {        
     