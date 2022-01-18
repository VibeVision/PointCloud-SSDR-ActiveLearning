#pragma once
#include <boost/config.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <utility>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>


using namespace std;
using namespace boost;

    typedef adjacency_list <vecS, vecS, undirectedS> Graph;
typedef typename graph_traits< Graph >::adjacency_iterator adjacency_iterator;

void connected_components(const uint32_t n_ver, const uint32_t n_edg
          , const uint32_t * Eu, const uint32_t * Ev, const char * active_edge
		  , std::vector<uint32_t> & in_component, std::vector< std::vector<uint32_t> > & components, const uint32_t cutoff)
{   //C-style interface

	Graph G(n_ver);
	for (uint32_t i_edg = 0; i_edg < n_edg; i_edg++)
	{
		if (active_edge[i_edg] > 0)
		{
			add_edge(Eu[i_edg], Ev[i_edg], G);
		}
	}
	
	int n_com = connected_components(G, &in_component[0]);

	//cout << "Total number of components: " << n_com << endl;
	
	std::vector< std::vector<uint32_t> > components_ssdr(n_com);
	for (uint32_t i_ver = 0; i_ver < n_ver; i_ver++)
	{
		components_ssdr[in_component[i_ver]].push_back(i_ver);
	}	

	//fuse components to preserve cutoff

	G = Graph(n_ver);
	for (uint32_t i_edg = 0; i_edg < n_edg; i_edg++)
	{
		if (active_edge[i_edg] == 0)
		{
			add_edge(Eu[i_edg], Ev[i_edg], G);
		}
	}

	typename graph_traits < Graph >::adjacency_iterator nei_ini, nei_end;
	boost::property_map<Graph, boost::vertex_index_t>::type vertex_index_map = get(boost::vertex_index, G);
	std::vector<int> is_fused(n_ver, 0);

	int n_com_final = n_com;
	for (int i_com = 0; i_com < n_com; i_com++)
	{
		if (components_ssdr[i_com].size() < cutoff)
		{//components is too small
			//std::cout << i_com << " of size " << components_ssdr[i_com].size() << " / " << cutoff << std::endl;
			int largest_neigh_comp_value = 0;
			int largest_neigh_comp_index = -1;
			for (int i_ver_com = 0; i_ver_com < components_ssdr[i_com].size(); i_ver_com++)
			{	//std::cout << "	considering node" << components_ssdr[i_com][i_ver_com] << std::endl;
				boost::tie(nei_ini, nei_end) = adjacent_vertic