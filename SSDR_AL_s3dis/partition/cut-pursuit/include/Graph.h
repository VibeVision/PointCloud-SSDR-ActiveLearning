#pragma once
#include "Common.h"
#include <boost/graph/properties.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graph_traits.hpp>

namespace CP {

typedef boost::graph_traits<boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> >::edge_descriptor
    EdgeDescriptor;

template <typename T> class VertexAttribute
{
public:
    typedef T calc_type;
public:
    T weight; //weight of the observation
    std::vector<T> observation; //observed value
    std::vector<T> value; //current value
    uint32_t color; //field use for the graph cut
    bool isBorder; //is the node part of an activated edge
    uint32_t in_component; //index of the component in which the node belong
public:
    VertexAttribute(uint32_t dim = 1, T weight=1.)
        :weight(weight), observation(dim,0.),value(dim,0.),color(-1)
        ,isBorder(false){}
};

template <typename T> class EdgeAttribute
{
public:
    typedef T calc_type;
public:
    uint32_t index; //index