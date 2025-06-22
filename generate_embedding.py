import torch
from module3 import Encoder
import pandas as pd
import os
#5
def load_model(model_path, device, in_channel, bottle_size, attn_drop, ffd_drop, hid_units, n_heads):

    model = Encoder(in_channel=in_channel, bottle_size=bottle_size, attn_drop=attn_drop, ffd_drop=ffd_drop,
                    hid_units=hid_units, n_heads=n_heads, residual=True).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['encoder_state_dict'])
    return model

def load_region_data(region):
    region_dir = os.path.join("data/graph", region)


    data_path = os.path.join(region_dir, f'{region}_data.pt')
    data = torch.load(data_path)
    data = data.x
    data = torch.tensor(data)

  
    edge_index_path = os.path.join(region_dir, f'{region}_edge.pt')
    edge_index_list = torch.load(edge_index_path)


    return data, edge_index_list

def save_embeddings_to_csv(embeddings, output_csv_filename):

    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df.to_csv(output_csv_filename, index=False, lineterminator='\n')

def extract_features(encoder, data, edge_index_list, device, output_csv_filename):
    
    encoder.eval()
    data = data.to(device)
    edge_index_list = [edge_index.to(device) for edge_index in edge_index_list]

    with torch.no_grad():
        bottle_vec = encoder(data, edge_index_list)

    save_embeddings_to_csv(bottle_vec.cpu(), output_csv_filename)

def process_csv_file(encoder, region, device, output_folder):
    data, edge_index_list = load_region_data(region)
  
    base_name = os.path.basename(f'{region}.csv')
    output_csv_filename = os.path.join(output_folder, f"embedding_{base_name}")

    extract_features(encoder, data, edge_index_list, device, output_csv_filename)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'run/model/region(10,16)_0.0244.pt'

    bottle_size = 15
    attn_drop = 0.3
    ffd_drop = 0.3
    en_hid_units = [300,300,300]
    en_n_heads = [2,2,2]
    in_channel = 500

  
    encoder = load_model(model_path, device, in_channel, bottle_size, attn_drop, ffd_drop, en_hid_units, en_n_heads)

    data_filename = 'data/data/regions/region_class.csv'
    regions = pd.read_csv(data_filename, usecols=[0]).iloc[:, 0]
    output_folder = 'run/embedding'

    os.makedirs(output_folder, exist_ok=True) 
    for region in regions:
        process_csv_file(encoder, region, device, output_folder)

if __name__ == '__main__':
    main()
