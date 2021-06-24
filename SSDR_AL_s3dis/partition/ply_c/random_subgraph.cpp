#pragma once
#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <boost/graph/adjacency_list.hpp>
#include <queue>

using namespace std;
using namespace boost;

namespace subgraph {

typedef adjacency_list <vecS, vecS, undirectedS> Graph;

typedef typename  boost::graph_traits< Graph >::vertex_descriptor VertexDescriptor;

typedef typename boost::property_map< Graph, boost::vertex_index_t>::type VertexIndexMap;

typedef typename graph_traits < Graph >::adjacency_iterator Adjacency_iterator;


void random_subgraph(const int n_ver, const int n_edg, const uint32_t * Eu, const uint32_t * Ev, int subgraph_size
                     , uint8_t * selected_edges,  uint8_t * selected_vertices)

{   //C-style interface

        if (n_ver < subgraph_size)
        {
            for (uint32_t i_edg = 0; i_edg < n_edg; i_edg++)
            {
                selected_edges[i_edg] = 1;
    