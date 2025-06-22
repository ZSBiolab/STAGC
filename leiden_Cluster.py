import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
import os

def build_knn_subgraph(data, n_neighbors):
   
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(data)
    distances, indices = knn.kneighbors(data)

    edges = [(i, indices[i, j])
             for i in range(data.shape[0])
             for j in range(1, n_neighbors)]

    return edges


graph_file_path = 'run/knn_graph.graphml'


data = pd.read_csv('data/data/umap_output.csv')


if os.path.exists(graph_file_path):

    g = ig.Graph.Read_GraphML(graph_file_path)
else:

    edges = build_knn_subgraph(data.iloc[:, 1:3].values, n_neighbors=40)


    g = ig.Graph()
    g.add_vertices(len(data))
    g.add_edges(edges)

 
    g.save(graph_file_path, format="graphml")


resolution_parameter = 0.03  


partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                     resolution_parameter=resolution_parameter)


clusters = partition.membership


num_clusters = len(set(clusters))
print(f"总共聚类出了 {num_clusters} 个类别。")


data['cluster_label'] = clusters


data.to_csv('run/updated_data.csv', index=False)
