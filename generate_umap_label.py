import pandas as pd
import os
import torch


region_class_df = pd.read_csv('data/data/regions/region_class.csv')
regions = region_class_df.iloc[:, 0] 


for region in regions:

    region_df = pd.read_csv(f"data/data/regions/{region}.csv")
    region_indices = region_df.iloc[:, 0] 

    
    umap_df = pd.read_csv('data/data/umap_output_with_centers_and_labels.csv')


    filtered_umap_df = umap_df[umap_df.iloc[:, 0].isin(region_indices)]


    os.makedirs(f"data/graph/{region}", exist_ok=True)

    
    filtered_umap_df.to_csv(f"data/graph/{region}/{region}_umap_label.csv", index=False)

 
    tensor_data = torch.tensor(filtered_umap_df.iloc[:, 3:5].values)
   
    print(f"Processing region: {region}")
    print(f"Columns being converted to tensor: {filtered_umap_df.columns[3]}, {filtered_umap_df.columns[4]}")

  
    torch.save(tensor_data, f"data/graph/{region}/{region}_umap_label.pt")
