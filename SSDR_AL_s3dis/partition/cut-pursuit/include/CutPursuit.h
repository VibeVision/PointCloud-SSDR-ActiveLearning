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
            addDoubledge<T>(this->main_graph, this->source, boost::vertex(ind_ver, this->main_graph), 0.,
                         eIndex, edge_attribute_map , false);
            eIndex +=2;
            addDoubledge<T>(this->main_graph, boost::vertex(ind_ver, this->main_graph), this->sink, 0.,
                         eIndex, edge_attribute_map, false);
            eIndex +=2;
            ++ite_ver;
        }

    }
    //=============================================================================================
    //================================  COMPUTE_REDUCE_VALUE  ====================================
    //=============================================================================================
    void compute_reduced_value()
    {
        for (uint32_t ind_com = 0;  ind_com < this->components.size(); ++ind_com)
        {   //compute the reduced value of each component
            compute_value(ind_com);
        }
    }
    //=============================================================================================
    //=============================   ACTIVATE_EDGES     ==========================================
    //=============================================================================================
    uint32_t activate_edges(bool allows_saturation = true)
    {   //this function analyzes the optimal binary partition to detect:
        //- saturated components (i.e. uncuttable)
        //- new activated edges
        VertexAttributeMap<T> vertex_attribute_map
            = boost::get(boost::vertex_bundle, this->main_graph);
        EdgeAttributeMap<T> edge_attribute_map
            = boost::get(boost::edge_bundle, this->main_graph);
        //saturation is the proportion of nodes in saturated components
        uint32_t saturation = 0;
        uint32_t nb_comp = this->components.size();
        //---- first check if the component are saturated-------------------------
        //#pragma omp parallel for if (this->parameter.parallel) schedule(dynamic)
        for (uint32_t ind_com = 0; ind_com < nb_comp; ind_com++)
        {
            if (this->saturated_components[ind_com])
            {   //ind_com is saturated, we increement saturation by ind_com size
                saturation += this->components[ind_com].size();
                continue;
            }
            std::vector<T> totalWeight(2,0);
            for (uint32_t ind_ver = 0;  ind_ver < this->components[ind_com].size(); ind_ver++)
            {
                bool isSink
                        = (vertex_attribute_map(this->components[ind_com][ind_ver]).color
                        ==  vertex_attribute_map(this->sink).color);
                if (isSink)
                {
                    totalWeight[0] += vertex_attribute_map(this->components[ind_com][ind_ver]).weight;
                }
                else
                {
                   totalWeight[1] += vertex_attribute_map(this->components[ind_com][ind_ver]).weight;
                }
            }
            if (allows_saturation && ((totalWeight[0] == 0)||(totalWeight[1] == 0)))
            {
                //the component is saturated
                this->saturateComponent(ind_com);
                saturation += this->components[ind_com].size();
            }
        }
        //----check which edges have been activated----
        EdgeIterator<T> ite_edg, ite_edg_end;
        uint32_t color_v1, color_v2, color_combination;
        for (boost::tie(ite_edg, ite_edg_end) = boost::edges(this->main_graph);
             ite_edg != ite_edg_end; ++ite_edg)
        {
            if (!edge_attribute_map(*ite_edg).realEdge )
            {
                continue;
            }
            color_v1 = vertex_attribute_map(boost::source(*ite_edg, this->main_graph)).color;
            color_v2 = vertex_attribute_map(boost::target(*ite_edg, this->main_graph)).color;
            //color_source = 0, color_sink = 4, uncolored = 1
            //we want an edge when a an interface source/sink
            //this corresponds to a sum of 4
            //for the case of uncolored nodes we arbitrarily chose source-uncolored
            color_combination = color_v1 + color_v2;
            if ((color_combination == 0)||(color_combination == 2)||(color_combination == 2)
              ||(color_combination == 8))
            {   //edge between two vertices of the same color
                continue;
            }
            //the edge is active!
            edge_attribute_map(*ite_edg).isActive = true;
            edge_attribute_map(*ite_edg).capacity = 0;
            vertex_attribute_map(boost::source(*ite_edg, this->main_graph)).isBorder = true;
            vertex_attribute_map(boost::target(*ite_edg, this->main_graph)).isBorder = true;
        }
        return saturation;
    }

    //=============================================================================================
    //=============================        REDUCE       ===========================================
    //=============================================================================================
    void reduce()
    {   //compute the reduced graph, and if need be performed a backward check
        this->compute_connected_components();
        if (this->parameter.backward_step)
        {   //compute the structure of the reduced graph        
            this->compute_reduced_graph();
            //check for beneficial merges
            this->merge(false);
        }
        else
        {   //compute only the value associated to each connected components
            this->compute_reduced_value();
        }
    }
    //=============================================================================================
    //==============================  compute_connected_components=========================================
    //=============================================================================================
    void compute_connected_components()
    {  //this function compute the connected components of the graph with active edges removed
        //the boolean vector indicating wether or not the edges and vertices have been seen already
        //the root is the first vertex of a component
        //this function is written such that the new components are appended at the end of components
        //this allows not to recompute saturated component
        VertexAttributeMap<T> vertex_attribute_map
            = boost::get(boost::vertex_bundle, this->main_graph);
        VertexIndexMap<T> vertex_index_map =get(boost::vertex_index, this->main_graph);
        //indicate which edges and nodes have been seen already by the dpsearch
        std::vector<bool> edges_seen (this->nEdge, false);
        std::vector<bool> vertices_seen (this->nVertex+2, false);
        vertices_seen[vertex_index_map(this->source)] = true;
        vertices_seen[vertex_index_map(this->sink)]   = true;
        //-------- start with the known roots------------------------------------------------------
        //#pragma omp parallel for if (this->parameter.parallel) schedule(dynamic)
        for (uint32_t ind_com = 0; ind_com < this->root_vertex.size(); ind_com++)
        {
            VertexDescriptor<T> root = this->root_vertex[ind_com]; //the first vertex of the component
            if (this->saturated_components[ind_com])
            {   //this component is saturated, we don't need to recompute it
                for (uint32_t ind_ver = 0; ind_ver < this->components[ind_com].size(); ++ind_ver)
                {
                    vertices_seen[vertex_index_map(this->components[ind_com][ind_ver])] = true;
                }
            }
            else
            {   //compute the new content of this component
                this->components.at(ind_com) = connected_comp_from_root(root, this->components.at(ind_com).size()
                                          , vertices_seen , edges_seen);
             }
        }
        //----now look for components that did not already exists----
        VertexIterator<T> ite_ver;
        for (ite_ver = boost::vertices(this->main_graph).first;
             ite_ver != this->lastIterator; ++ite_ver)
        {
            if (vertices_seen[vertex_index_map(*ite_ver)])
            {
                 continue;
            } //this vertex is not currently in a connected component
            VertexDescriptor<T> root = *ite_ver; //we define it as the root of a new component
            uint32_t current_component_size =
                    this->components[vertex_attribute_map(root).in_component].size();
            this->components.push_back(
                    connected_comp_from_root(root, current_component_size
                  , vertices_seen, edges_seen));
            this->root_vertex.push_back(root);
            this->saturated_components.push_back(false);
        }
        this->components.shrink_to_fit();
    }
    //=============================================================================================
    //==============================  CONNECTED_COMP_FROM_ROOT=========================================
    //=============================================================================================
    inline std::vector<VertexDescriptor<T>> connected_comp_from_root(const VertexDescriptor<T> & root
                , const uint32_t & size_comp, std::vector<bool> & vertices_seen , std::vector<bool> & edges_seen)
    {
        //this function compute the connected component of the graph with active edges removed
        // associated with the root ROOT by performing a depth search first
        EdgeAttributeMap<T> edge_attribute_map
             = boost::get(boost::edge_bundle, this->main_graph);
        VertexIndexMap<T> vertex_index_map = get(boost::vertex_index, this->main_graph);
        EdgeIndexMap<T>   edge_index_map = get(&EdgeAttribute<T>::index, this->main_graph);
        std::vector<VertexDescriptor<T>> vertices_added; //the vertices in the current connected component
        // vertices_added contains the vertices that have been added to the current coomponent
        vertices_added.reserve(size_comp);
        //heap_explore contains the vertices to be added to the current component
        std::vector<VertexDescriptor<T>> vertices_to_add;
        vertices_to_add.reserve(size_comp);
        VertexDescriptor<T> vertex_current; //the node being consideed
        EdgeDescriptor      edge_current, edge_reverse; //the edge being considered
        //fill the heap with the root node
        vertices_to_add.push_back(root);
        while (vertices_to_add.size()>0)
        {   //as long as there are vertices left to add
            vertex_current = vertices_to_add.back(); //the current node is the last node to add
            vertices_to_add.pop_back(); //remove the current node from the vertices to add
            if (vertices_seen[vertex_index_map(vertex_current)])
            {   //this vertex has already been treated
                continue;
            }
            vertices_added.push_back(vertex_current); //we add the current node
            vertices_seen[vertex_index_map(vertex_current)] = true ; //and flag it as seen
            //----we now explore the neighbors of current_node
            typename boost::graph_traits<Graph<T>>::out_edge_iterator ite_edg, ite_edg_end;
            for (boost::tie(ite_edg,ite_edg_end) = boost::out_edges(vertex_current, this->main_graph);
                ite_edg !=  ite_edg_end; ++ite_edg)
                {   //explore edges leaving current_node
                    edge_current = *ite_edg;
                    if (edge_attribute_map(*ite_edg).isActive || (edges_seen[edge_index_map(edge_current)]))
                    {   //edge is either active or treated, we skip it
                        continue;
                    }
                    //the target of this edge is a node to add
                    edge_reverse = edge_attribute_map(edge_current).edge_reverse;
                    edges_seen[edge_index_map(edge_current)] = true;
                    edges_seen[edge_index_map(edge_reverse)] = true;
                    vertices_to_add.push_back(boost::target(edge_current, this->main_graph));
               }
            }
            vertices_added.shrink_to_fit();
            return vertices_added;
    }
    //======================================================================================