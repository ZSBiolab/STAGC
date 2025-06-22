import os
import cudf
import cuml
from cuml.decomposition import PCA
from cuml.manifold import UMAP
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


file_path = 'DLPFC2/normalized_data_final.csv'
df = cudf.read_csv(file_path, usecols=[i for i in range(1, 16604+1)]) 


pca = PCA(n_components=25)
df_pca = pca.fit_transform(df)


umap = UMAP(n_components=2)
df_umap = umap.fit_transform(df_pca)


print("df_umap columns (cuDF):", df_umap.columns)


first_column = cudf.read_csv(file_path).iloc[:, 0]
umap_result_with_first_column = cudf.DataFrame({
    'first_column': first_column,
    'UMAP_1': df_umap.iloc[:, 0],
    'UMAP_2': df_umap.iloc[:, 1]
})


df_umap_host = df_umap.to_pandas()


graph_file = 'DLPFC2/graph.pickle'
if os.path.exists(graph_file):
    
    g = ig.Graph.Read_Pickle(graph_file)
else:
   
    n_neighbors = 15
    knn_graph = cuml.neighbors.NearestNeighbors(n_neighbors=n_neighbors)
    knn_graph.fit(df_umap)
    knn_indices = knn_graph.kneighbors(df_umap, return_distance=False).values_host

   
    edges = [(int(i), int(j)) for i in range(len(knn_indices)) for j in knn_indices[i]]
    g = ig.Graph(edges)

    g.write_pickle(graph_file)


resolution_parameter = 0.15
partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=resolution_parameter)
clusters = np.array(partition.membership)


df_umap_host['cluster'] = clusters


cluster_centers = df_umap_host.groupby('cluster').mean()


cluster_mapping_x = df_umap_host['cluster'].map(cluster_centers.iloc[:, 0])
cluster_mapping_y = df_umap_host['cluster'].map(cluster_centers.iloc[:, 1])


umap_result_with_first_column['center_x'] = cudf.Series(cluster_mapping_x.values)
umap_result_with_first_column['center_y'] = cudf.Series(cluster_mapping_y.values)


umap_result_with_first_column['cluster_label'] = cudf.Series(df_umap_host['cluster'].values)  # Add cluster labels to the cuDF DataFrame
umap_output_file = 'DLPFC2/umap_output_with_centers_and_labels.csv'  # Update the file name
umap_result_with_first_column.to_csv(umap_output_file, index=False)

print(f"UMAP dimension reduction result, cluster center and cluster label have been saved as: {umap_output_file}")


umap_1_numpy = umap_result_with_first_column['UMAP_1'].to_numpy()
umap_2_numpy = umap_result_with_first_column['UMAP_2'].to_numpy()

plt.figure(figsize=(10, 8))
plt.scatter(umap_1_numpy, umap_2_numpy, c=df_umap_host['cluster'], cmap='viridis', s=1)
plt.title('vis')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.colorbar(label='Cluster ID')
plt.scatter(cluster_centers.iloc[:, 0], cluster_centers.iloc[:, 1], c='red', s=50, marker='x')
for i, row in cluster_centers.iterrows():
    plt.text(row[0], row[1], str(i), color='black', fontsize=12)

plt.show()
