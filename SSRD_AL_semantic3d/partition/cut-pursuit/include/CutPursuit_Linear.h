#pragma once
#include "Common.h"
#include "CutPursuit.h"

namespace CP {

template <typename T>
class CutPursuit_Linear : public CutPursuit<T>
{
    public:
   ~CutPursuit_Linear(){
    };        
   // virtual ~CutPursuit_Linear();  
    std::vector<std::vector<T>> componentVector;
        // only used with backward step - the sum of all observation in the component
    CutPursuit_Linear(uint32_t nbVertex = 1) : CutPursuit<T>(nbVertex)
    {
        this->componentVector  = std::vector<std::vector<T>>(1);
    }

    virtual std::pair<T,T> compute_energy() override
    {
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->main_graph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->main_graph);
        std::pair<T,T> pair_energy;
        T energy = 0;
        VertexIterator<T> i_ver;
        //#pragma omp parallel for private(i_ver) if (this->parameter.parallel)
        for (i_ver = boost::vertices(this->main_graph).first;
             i_ver != this->lastIterator; ++i_ver)
        {
            for(uint32_t i_dim=0; i_dim<this->dim; i_dim++)
            {
                energy -= vertex_attribute_map(*i_ver).weight
                        * vertex_attribute_map(*i_ver).observation[i_dim]
                        * vertex_attribute_map(*i_ver).value[i_dim];
            }
        }
        pair_energy.first = energy;
        energy = 0;
        EdgeIterator<T> i_edg, i_edg_end =  boost::edges(this->main_graph).second;
        for (i_edg = boost::edges(this->main_graph).first;
             i_edg != i_edg_end; ++i_edg)
        {
            if (!edge_attribute_map(*i_edg).realEdge)
            {
                continue;
            }
            energy += .5 * edge_attribute_map(*i_edg).isActive * this->parameter.reg_strenth
                    * edge_attribute_map(*i_edg).weight;
        }
        pair_energy.second = energy;
        return pair_energy;
    }

    //=============================================================================================
    //=============================        SPLIT        ===========================================
    //=============================================================================================
    virtual uint32_t split()
    { // split the graph by trying to find the best binary partition
      // each components is split into B and notB
        uint32_t saturation;
        //initialize h_1 and h_2 with kmeans
        //--------initilializing labels------------------------------------------------------------
        //corner contains the two most likely class for each component
        std::vector< std::vector< uint32_t > > corners =
                std::vector< std::vector< uint32_t > >(this->components.size(),
                std::vector< uint32_t >(2,0));
        this->compute_corners(corners);
        this->set_capacities(corners);
        //compute flow
        boost::boykov_kolmogorov_max_flow(
                   this->main_graph,
                   get(&EdgeAttribute<T>::capacity        , this->main_graph),
                   get(&EdgeAttribute<T>::residualCapacity, this->main_graph),
                   get(&EdgeAttribute<T>::edge_reverse     , this->main_graph),
                   get(&VertexAttribute<T>::color         , this->main_graph),
                   get(boost::vertex_index                , this->main_graph),
                   this->source,
                   this->sink);
        saturation = this->activate_edges();
        return saturation;
    }
    //=============================================================================================
    //=============================      COMPUTE CORNERS        ===================================
    //=============================================================================================
    inline void compute_corners(std::vector< std::vector< uint32_t > > & corners)
    { //-----compute the 2 most populous labels------------------------------
         //#pragma omp parallel for if (this->parameter.parallel) schedule(dynamic)
        for (uint32_t  i_com =0;i_com < this->components.size(); i_com++)
        {
            if (this->saturated_components[i_com])
            {
                continue;
            }
            std::pair<uint32_t, uint32_t> corners_pair = find_corner(i_com);
            corners[i_com][0] = corners_pair.first;
            corners[i_com][1] = corners_pair.second;
        }
        return;
    }
    //=============================================================================================
    //=============================     find_corner        =======================================
    //=============================================================================================
    std::pair<uint32_t, uint32_t> find_corner(const uint32_t & i_com)
    {
        // given a component will output the pairs of the two most likely labels
        VertexAttributeMap<T> vertex_attribute_map
                                    = boost::get(boost::vertex_bundle, this->main_graph);
        std::vector<T> average_vector(this->dim,0);
        for (uint32_t i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
        {
            for(uint32_t i_dim=0; i_dim < this->dim; i_dim++)
            {
            average_vector.at(i_dim) += vertex_attribute_map[this->components[i_com][i_ver]].observation[i_dim]
                                *  vertex_attribute_map[this->components[i_com][i_ver]].weight;
