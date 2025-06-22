import pandas as pd
import numpy as np
import os
import shutil
from sklearn.neighbors import NearestNeighbors
import random
from torch_geometric.data import Data, InMemoryDataset
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.decomposition import PCA

feature = 500


def combined_similarity(vector_a, vector_b, cos_weight=0.5, euc_weight=0.5, max_euc_dist=100):
 
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    cos_sim = dot_product / (norm_a * norm_b)

    
    euc_dist = np.linalg.norm(vector_a - vector_b)
    adjusted_euc_sim = -1 * (euc_dist - max_euc_dist) / max_euc_dist

    
    return cos_weight * cos_sim + euc_weight * adjusted_euc_sim


def generate_edges(df, k_nn, k_sim, region, output_dir):

    if df.shape[0] <= k_nn:
        print(f"Skipping {region} due to insufficient columns: {df.shape[1]} columns")
        return

    if df.columns[feature + 1] != 'x' or df.columns[feature + 2] != 'y':
        raise ValueError("'x' and 'y' documents are wrong.")

   
    region_folder = output_dir

    
    shutil.copy(f"data/data/regions/{region}.csv", os.path.join(region_folder, f"{region}.csv"))

  
    gene_expressions = df.iloc[:, 1:feature + 1].to_numpy()

    gene_expressions_pca = gene_expressions

    x = df.iloc[:, feature + 1].to_numpy()
    y = df.iloc[:, feature + 2].to_numpy()
    num_nodes = len(gene_expressions)
 
    nbrs = NearestNeighbors(n_neighbors=k_nn, algorithm='ball_tree').fit(np.column_stack((x, y)))


    edges = defaultdict(list)


    for i in range(len(df)):
        selected_indices = [i]
        current_mean = gene_expressions_pca[i]

   
        for _ in range(k_sim):
         
            distances, indices = nbrs.kneighbors([np.column_stack((x[i], y[i]))[0]])
            indices = indices[0]

         
            filtered_indices = [idx for idx in indices if idx != i and df.iloc[idx, 0] not in selected_indices]

           
            similarities = [combined_similarity(current_mean, gene_expressions_pca[idx]) for idx in filtered_indices]

          
            if similarities:
                top_indices = np.argsort(similarities)[-5:]  
                chosen_idx = random.choice(top_indices) 
                selected_idx = filtered_indices[chosen_idx]

                selected_indices.append(selected_idx)
                edges['st_st'].append((i, selected_idx))

              
                current_mean = np.mean([current_mean, gene_expressions_pca[selected_idx]], axis=0)


    print(f"Total edges for {region} before deduplication: {len(edges['st_st'])}")
    if 'st_st' in edges:
    
        edges['st_st'] = list(set(edges['st_st']))
    print(f"Total edges for {region} after deduplication: {len(edges['st_st'])}")
  
    edge_index_list = []

    for edge_type, edge_list in edges.items():
   
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_index_list.append(edge_index)
    region_dir = os.path.join(region_folder, region)

    torch.save(edge_index_list, os.path.join(region_dir, f'{region}_edge.pt'))

    #torch.save(label_matrix, os.path.join(region_dir, f'{region}_label_matrix.pt'))


class HeteroGraphDataset(InMemoryDataset):
    def __init__(self, filename, k_nn=20, k_sim=5, transform=None, output_dir="data/graph"):
        self.filename = filename
        self.k_nn = k_nn
        self.k_sim = k_sim
        self.output_dir = output_dir
        super(HeteroGraphDataset, self).__init__(output_dir, transform)

       
        region_output_dir = os.path.join(output_dir, "region(1,9)")
        region_data_file = os.path.join(region_output_dir, "region(1,9)_data.pt")

  
        if not os.path.exists(region_data_file):
            print("Data not found, processing...")
            os.makedirs(region_output_dir, exist_ok=True)
            self.process_data(self.filename, self.k_nn, self.k_sim)
        else:
            print("Data already exists, skipping processing")

    def process_data(self, filename, k_nn, k_sim):
        regions = pd.read_csv(filename, usecols=[0]).iloc[:, 0]

     
        for region in tqdm(regions, desc='Processing regions'):
            region_df = pd.read_csv(f"data/data/regions/{region}.csv")

            gene_expressions = region_df.iloc[:, 1:feature + 1].values
        
            try:
                st_node_features = torch.tensor(gene_expressions, dtype=torch.float)
            except TypeError as e:
                print(f"error, region: {region}：{e}")
                print("gene_expressions data type：", gene_expressions.dtype)
                print("gene_expressions ：", gene_expressions)

            num_nodes = len(st_node_features)
            print(f"Total num_nodes of {region}: {num_nodes}")
            max_dim = gene_expressions.shape[1]
            inputs = torch.zeros((num_nodes, max_dim))
            inputs[:len(st_node_features), :gene_expressions.shape[1]] = st_node_features
            data = Data(num_nodes=num_nodes)
            data.x = inputs
            region_dir = os.path.join(self.output_dir, region)
            if not os.path.exists(region_dir):
                os.makedirs(region_dir)
            torch.save(data, os.path.join(region_dir, f'{region}_data.pt'))

            # Call generate_edges function
            generate_edges(region_df, k_nn, k_sim, region, output_dir=self.output_dir)






