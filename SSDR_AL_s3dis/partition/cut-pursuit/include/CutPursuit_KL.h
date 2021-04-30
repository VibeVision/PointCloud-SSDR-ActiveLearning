#pragma once
#include "Common.h"
#include "CutPursuit.h"

namespace CP {
template <typename T>
class CutPursuit_KL : public CutPursuit<T>
{
    public:
    ~CutPursuit_KL(){
    };      
    virtual std::pair<T,T> compute_energy() override
    {
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->main_graph);
        EdgeAttributeMap<T> edge_attribute_map
                = boost::get(boost::edge_bundle, this->main_graph);
        std::pair<T,T> pair_energy;
        T energy = 0, smoothedObservation, smoothedValue;
        //#pragma omp parallel if (this->parameter.parallel)
        for (VertexIterator<T> i_ver = boost::vertices(this->main_graph).first;
                     i_ver != this->lastIterator; ++i_ver)
        {
            for(uint32_t  i_dim=0; i_dim<this->dim; i_dim++)
            { //smoothing as a linear combination with the uniform probability
            smoothedObservation =
            	this->parameter.smoothing / this->dim
                + (1 - this->parameter.smoothing)
                * vertex_attribute_map(*i_ver).observation[i_dim];
            smoothedValue =
            	this->parameter.smoothing / this->dim
                + (1 - this->parameter.smoothing)
                * vertex_attribute_map(*i_ver).value[i_dim];
            energy += smoothedObservation
                * (log(smoothedObservation) - log(smoothedValue))
                * vertex_attribute_map(*i_ver).weight;
            }
        }
        pair_energy.first = energy;
        energy = 0;
        EdgeIterator<T> i_edg_end =  boost::edges(this->main_graph).second;
       for (EdgeIterator<T> i_edg = boost::edges(this->main_graph).first;
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
    virtual uint32_t  split() override
    { // split the graph by trying to find the best binary partition
      // each compone