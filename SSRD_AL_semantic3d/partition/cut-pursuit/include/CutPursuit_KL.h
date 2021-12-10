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
      // each components is split into B and notB
      // for each components we associate the value h_1 and h_2 to vertices in B or notB
      // the affectation as well as h_1 and h_2 are computed alternatively
      //tic();
      //--------loading structures---------------------------------------------------------------
        TimeStack ts; ts.tic();
        uint32_t  nb_comp = this->components.size();
        VertexAttributeMap<T> vertex_attribute_map
                   = boost::get(boost::vertex_bundle, this->main_graph);
        VertexIndexMap<T> vertex_index_map = boost::get(boost::vertex_index, this->main_graph);
        uint32_t  saturation;
        //initialize h_1 and h_2 with kmeans
        //stores wether each vertex is B or notB
        std::vector<bool> binary_label(this->nVertex);
        this->init_labels(binary_label);
        VectorOfCentroids<T> centers(nb_comp, this->dim);
        //-----main loop----------------------------------------------------------------
                // the optimal flow is iteratively approximated
        for (uint32_t  i_step = 1; i_step <= this->parameter.flow_steps; i_step++)
        {
            //compute h_1 and h_2
            centers = VectorOfCentroids<T>(nb_comp, this->dim);
            this->compute_centers(centers, binary_label);
            // update the capacities of the flow graph
            this->set_capacities(centers);
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
            for (uint32_t  i_com = 0; i_com < nb_comp; i_com++)
            {
                if (this->saturated_components[i_com])
                {
                    continue;
                }
                for (uint32_t  i_ver = 0;  i_ver < this->components[i_com].size(); i_ver++)
                {
                    binary_label[vertex_index_map(this->components[i_com][i_ver])]
                          = (vertex_attribute_map(this->components[i_com][i_ver]).color
                          == vertex_attribute_map(this->sink).color);
                 }
             }
        }
        saturation = this->activate_edges();
        return saturation;
    }
    //=============================================================================================
    //=============================    INIT_KL  ===================================================
    //=============================================================================================
    inline void init_labels(std::vector<bool> & binary_label)
    { //-----initialize the labelling for each components with kmeans------------------------------
        VertexAttributeMap<T> vertex_attribute_map
                = boost::get(boost::vertex_bundle, this->main_graph);
        VertexIndexMap<T> vertex_index_map = boost::get(boost::vertex_index, this->main_graph);
        std::vector< std::vector<T> > kernels(2, std::vector<T>(this->dim));
        std::vector< std::vector<T> > smooth_kernels(2, std::vector<T>(this->dim));
        T total_weight[2];
        uint32_t  nb_comp = this->components.size();
        T best_energy, current_energy;
        //#pragma omp parallel for private(kernels, total_weight, best_energy, current_energy) if (this->parameter.parallel && nb_comp>8) schedule(dynamic)
        for (uint32_t  i_com = 0; i_com < nb_comp; i_com++)
        {
            uint32_t  comp_size = this->components[i_com].size();
            std::vector<bool> potential_label(comp_size);    
            std::vector<T> energy_array(comp_size);
            std::vector<T> constant_part(comp_size);
            std::vector< std::vector<T> > smooth_obs(comp_size, std::vector<T>(2,0));
            if (this->saturated_components[i_com] || comp_size <= 1)
            {
                continue;
            }
            //KL fidelity has a part that depends
            //purely on the observation that can be precomputed
            //#pragma omp parallel for if (this->parameter.parallel && nb_comp<=8) schedule(dynamic)
            for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
            {
                constant_part[i_ver] = 0;
                for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                {
                    smooth_obs[i_ver][i_dim] = 0;
                }
                for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                {
                    smooth_obs[i_ver][i_dim] =
                        this->parameter.smoothing / this->dim
                        + (1 - this->parameter.smoothing)
                        * vertex_attribute_map(this->components[i_com][i_ver]).observation[i_dim];
                    constant_part[i_ver] += smooth_obs[i_ver][i_dim]
                           * log(smooth_obs[i_ver][i_dim]) 
                           * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                 }
            }
            for (uint32_t  init_kmeans = 0; init_kmeans < this->parameter.kmeans_resampling; init_kmeans++)
            {
                //----- initialization with KM++ ------------------
                // first kernel chosen randomly
                uint32_t  first_kernel  = std::rand() % comp_size, second_kernel = 0;
                for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                {   //fill the first kernel
                    kernels[0][i_dim] = vertex_attribute_map(this->components[i_com][first_kernel ]).observation[i_dim];
                    smooth_kernels[0][i_dim] =  this->parameter.smoothing
                        / this->dim + (1 - this->parameter.smoothing)
                        * kernels[0][i_dim];
                }
                //now compute the square distance of each pouint32_t to this kernel
                best_energy = 0; //energy total
                //#pragma omp parallel for if (this->parameter.parallel && nb_comp<=8) schedule(dynamic)
                for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                {
                    energy_array[i_ver] = constant_part[i_ver];
                    for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                    {
                         energy_array[i_ver] -=
                            smooth_obs[i_ver][i_dim]
                            * log(smooth_kernels[0][i_dim])
                            * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                    }
                    energy_array[i_ver] = pow(energy_array[i_ver],2);
                    best_energy += energy_array[i_ver];
                } // we now generate a random number to determinate which node will be the second kernel
                if (best_energy==0)
                { //all the points in this components are identical
                    for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                    {
                        binary_label[vertex_index_map(this->components[i_com][i_ver])] = false;
                    }
                   break;
                }
                //we now choose the second kernel with a probability
                //proportional to the square distance
                T random_sample = ((T)(rand())) / ((T)(RAND_MAX));
                current_energy = best_energy * random_sample;
                for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                {
                    current_energy -= energy_array[i_ver];
                    if (current_energy < 0)
                    { //we have selected the second kernel
                        second_kernel = i_ver;
                        break;
                    }
                }
                for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                { // now fill the second kernel
                   kernels[1][i_dim] = vertex_attribute_map(this->components[i_com][second_kernel]).observation[i_dim];
                   smooth_kernels[1][i_dim] =  this->parameter.smoothing
                            / this->dim + (1 - this->parameter.smoothing)
                            * kernels[1][i_dim];
                }
                //----main kmeans loop-----
                for (uint32_t  ite_kmeans = 0; ite_kmeans < this->parameter.kmeans_ite; ite_kmeans++)
                {
                    //--affectation step: associate each node with its closest kernel-------------------
                    //#pragma omp parallel for if (this->parameter.parallel && nb_comp<=8) schedule(dynamic)
                    for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                    {
                        //the distance to each kernel
                        std::vector<T> distance_kernels(2, constant_part[i_ver]);
                        for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                        {
                            distance_kernels[0] -= smooth_obs[i_ver][i_dim]
                                * log(smooth_kernels[0][i_dim])
                                * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                            distance_kernels[1] -= smooth_obs[i_ver][i_dim]
                                * log(smooth_kernels[1][i_dim])
                                * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                        }
                        potential_label[i_ver] = distance_kernels[0] > distance_kernels[1];
                    }
                    //-----computation of the new kernels----------------------------
                    total_weight[0] = 0.;
                    total_weight[1] = 0.;
                    for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                    {
                       kernels[0][i_dim] = 0;
                       kernels[1][i_dim] = 0;
                    }
                    for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                    {
                        if (vertex_attribute_map(this->components[i_com][i_ver]).weight==0)
                        {
                            continue;
                        }
                        if (potential_label[i_ver])
                        {
                            total_weight[0] += vertex_attribute_map(this->components[i_com][i_ver]).weight;
                            for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                            {
                                kernels[0][i_dim] += 
                                    vertex_attribute_map(this->components[i_com][i_ver]).observation[i_dim]
                                    * vertex_attribute_map(this->components[i_com][i_ver]).weight ;
                             }
                         }
                         else
                         {
                            total_weight[1] += vertex_attribute_map(this->components[i_com][i_ver]).weight;
                            for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                            {
                                kernels[1][i_dim] += 
                                    vertex_attribute_map(this->components[i_com][i_ver]).observation[i_dim]
                                    * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                            }
                         }
                    }
                    if ((total_weight[0] == 0)||(total_weight[1] == 0))
                    {
                        std::cout << "kmeans error" << std::endl;
                    }
                    for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                    {
                        kernels[0][i_dim] = kernels[0][i_dim] / total_weight[0];
                        kernels[1][i_dim] = kernels[1][i_dim] / total_weight[1];
                        smooth_kernels[0][i_dim] =  this->parameter.smoothing
                            / this->dim + (1 - this->parameter.smoothing)
                            * kernels[0][i_dim];
                        smooth_kernels[1][i_dim] =  this->parameter.smoothing
                            / this->dim + (1 - this->parameter.smoothing)
                            * kernels[1][i_dim];
                    }
                }
                //----compute the associated energy ------
                current_energy = 0;
                for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                {
                    current_energy += constant_part[i_ver];
                    if (potential_label[i_ver])
                    {
                        for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                        {
                            current_energy -= smooth_obs[i_ver][i_dim]
                                 * log(smooth_kernels[0][i_dim])
                                 * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                        }
                    }
                    else
                    {
                        for(uint32_t  i_dim=0; i_dim < this->dim; i_dim++)
                        {
                            current_energy -= smooth_obs[i_ver][i_dim]
                                 * log(smooth_kernels[1][i_dim])
                                 * vertex_attribute_map(this->components[i_com][i_ver]).weight;
                        }
                    }
                }
                if (current_energy < best_energy)
                {
                    best_energy = current_energy;
                    for (uint32_t  i_ver = 0;  i_ver < comp_size; i_ver++)
                    {
                        binary_label[vertex_index_map(this->components[i_com][i_ver])] = potential_label[i_ver];
                    }
              