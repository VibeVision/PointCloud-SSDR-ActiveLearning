
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Delaunay
from numpy import linalg as LA
import numpy.matlib
#------------------------------------------------------------------------------
def compute_graph_nn(xyz, k_nn):
    """compute the knn graph"""
    num_ver = xyz.shape[0]
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    source = np.matlib.repmat(range(0, num_ver), k_nn, 1).flatten(order='F')
    #save the graph
    graph["source"] = source.flatten().astype('uint32')
    graph["target"] = neighbors.flatten().astype('uint32')
    graph["distances"] = distances.flatten().astype('float32')
    return graph
#------------------------------------------------------------------------------
def compute_graph_nn_2(xyz, k_nn1, k_nn2, voronoi = 0.0):
    """compute simultaneously 2 knn structures
    only saves target for knn2
    assumption : knn1 <= knn2"""
    assert k_nn1 <= k_nn2, "knn1 must be smaller than knn2"
    n_ver = xyz.shape[0]
    #compute nearest neighbors
    graph = dict([("is_nn", True)])
    nn = NearestNeighbors(n_neighbors=k_nn2+1, algorithm='kd_tree').fit(xyz)
    distances, neighbors = nn.kneighbors(xyz)
    del nn
    neighbors = neighbors[:, 1:]
    distances = distances[:, 1:]
    #---knn2---
    target2 = (neighbors.flatten()).astype('uint32')
    #---knn1-----
    if voronoi>0:
        tri = Delaunay(xyz)
        graph["source"] = np.hstack((tri.vertices[:,0],tri.vertices[:,0], \
              tri.vertices[:,0], tri.vertices[:,1], tri.vertices[:,1], tri.vertices[:,2])).astype('uint64')
        graph["target"]= np.hstack((tri.vertices[:,1],tri.vertices[:,2], \
              tri.vertices[:,3], tri.vertices[:,2], tri.vertices[:,3], tri.vertices[:,3])).astype('uint64')
        graph["distances"] = ((xyz[graph["source"],:] - xyz[graph["target"],:])**2).sum(1)
        keep_edges = graph["distances"]<voronoi
        graph["source"] = graph["source"][keep_edges]
        graph["target"] = graph["target"][keep_edges]
        
        graph["source"] = np.hstack((graph["source"], np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')))
        neighbors = neighbors[:, :k_nn1]
        graph["target"] =  np.hstack((graph["target"],np.transpose(neighbors.flatten(order='C')).astype('uint32')))
        
        edg_id = graph["source"] + n_ver * graph["target"]
        
        dump, unique_edges = np.unique(edg_id, return_index = True)
        graph["source"] = graph["source"][unique_edges]
        graph["target"] = graph["target"][unique_edges]
       
        graph["distances"] = graph["distances"][keep_edges]
    else:
        neighbors = neighbors[:, :k_nn1]
        distances = distances[:, :k_nn1]
        graph["source"] = np.matlib.repmat(range(0, n_ver)
            , k_nn1, 1).flatten(order='F').astype('uint32')
        graph["target"] = np.transpose(neighbors.flatten(order='C')).astype('uint32')
        graph["distances"] = distances.flatten().astype('float32')
    #save the graph
    return graph, target2
#------------------------------------------------------------------------------
def compute_sp_graph(xyz, d_max, in_component, components, labels, n_labels):
    """compute the partition graph with superpoints and superedges features"""
    n_com = max(in_component)+1
    in_component = np.array(in_component)
    has_labels = len(labels) > 1
    label_hist = has_labels and len(labels.shape) > 1 and labels.shape[1] > 1
    #---compute delaunay triangulation---
    tri = Delaunay(xyz)
    #interface select the edges between different components
    #edgx and edgxr converts from tetrahedrons to edges
	#done separatly for each edge of the tetrahedrons to limit memory impact
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 1]]
    edg1 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 1]))
    edg1r = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 2]]
    edg2 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 2]))
    edg2r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 0]] != in_component[tri.vertices[:, 3]]
    edg3 = np.vstack((tri.vertices[interface, 0], tri.vertices[interface, 3]))
    edg3r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 0]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 2]]
    edg4 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 2]))
    edg4r = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 1]] != in_component[tri.vertices[:, 3]]
    edg5 = np.vstack((tri.vertices[interface, 1], tri.vertices[interface, 3]))
    edg5r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 1]))
    interface = in_component[tri.vertices[:, 2]] != in_component[tri.vertices[:, 3]]
    edg6 = np.vstack((tri.vertices[interface, 2], tri.vertices[interface, 3]))
    edg6r = np.vstack((tri.vertices[interface, 3], tri.vertices[interface, 2]))
    del tri, interface
    edges = np.hstack((edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r,
                       edg3r, edg4r ,edg5r, edg6r))
    del edg1, edg2, edg3, edg4 ,edg5, edg6, edg1r, edg2r, edg3r, edg4r, edg5r, edg6r
    edges = np.unique(edges, axis=1)
    
    if d_max > 0:
        dist = np.sqrt(((xyz[edges[0,:]]-xyz[edges[1,:]])**2).sum(1))
        edges = edges[:,dist<d_max]
	
    #---sort edges by alpha numeric order wrt to the components of their source/target---
    n_edg = len(edges[0])
    edge_comp = in_component[edges]
    edge_comp_index = n_com * edge_comp[0,:] +  edge_comp[1,:]
    order = np.argsort(edge_comp_index)
    edges = edges[:, order]
    edge_comp = edge_comp[:, order]
    edge_comp_index = edge_comp_index[order]
    #marks where the edges change components iot compting them by blocks
    jump_edg = np.vstack((0, np.argwhere(np.diff(edge_comp_index)) + 1, n_edg)).flatten()
    n_sedg = len(jump_edg) - 1
    #---set up the edges descriptors---
    graph = dict([("is_nn", False)])
    graph["sp_centroids"] = np.zeros((n_com, 3), dtype='float32')
    graph["sp_length"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_surface"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_volume"] = np.zeros((n_com, 1), dtype='float32')
    graph["sp_point_count"] = np.zeros((n_com, 1), dtype='uint64')
    graph["source"] = np.zeros((n_sedg, 1), dtype='uint32')
    graph["ta