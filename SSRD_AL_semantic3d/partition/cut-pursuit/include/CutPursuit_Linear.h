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
