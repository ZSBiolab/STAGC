import torch
import torch.optim as optim
from dataset import HeteroGraphDataset
from torch.optim.lr_scheduler import CosineAnnealingLR 
from sklearn.metrics import f1_score
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def write_log(message):
    with open('run/unsup.txt', 'a') as f:
        f.write(message + '\n')
    print(message)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_filename = 'data/data/regions/region_class.csv'
    HeteroGraphDataset(data_filename)


    bottle_size = 16
    attn_drop = 0
    ffd_drop = 0
    en_hid_units = [500,500,500]
    en_n_heads = [2,2,2]
    input = 161
    best_loss = float('inf')


    use_self_supervised_clustering = False

    def self_supervised_clustering(bottle_vec, cluster_centers, gamma):
        def compute_qiu(bottle_vec, cluster_centers):
            dist_squared = torch.sum((bottle_vec.unsqueeze(1) - cluster_centers) ** 2, dim=2)
            q_iu_numerator = (1 + dist_squared) ** -1
            q_iu_denominator = torch.sum(q_iu_numerator, dim=1, keepdim=True)
            q_iu = q_iu_numerator / q_iu_denominator
            return q_iu

        def compute_piu(q_iu):
            numerator = q_iu ** 2 / torch.sum(q_iu, dim=0, keepdim=True)
            denominator = torch.sum(numerator, dim=1, keepdim=True)
            p_iu = numerator / denominator
            return p_iu

        q_iu = compute_qiu(bottle_vec, cluster_centers)
        p_iu = compute_piu(q_iu)
        kl_loss = F.kl_div(q_iu.log(), p_iu, reduction='batchmean')
        return kl_loss * gamma

    if not os.path.exists('run/model'):
        os.makedirs('run/model')

    hyperparameters = f'''
            input: {input}
            nb_classes: {bottle_size}
            attn_drop: {attn_drop}
            ffd_drop: {ffd_drop}
            '''

    from module import Encoder, Decoder
    Encoder = Encoder(in_channel=input, bottle_size=bottle_size, attn_drop=attn_drop, ffd_drop=ffd_drop,
                         hid_units=en_hid_units, n_heads=en_n_heads, residual=True).to(device)
    Decoder = Decoder(in_channel = bottle_size , out_channel=input).to(device)

    optimizer = optim.Adam(list(Encoder.parameters()) + list(Decoder.parameters()), lr=0.002, weight_decay=0.00001)
    criterion = torch.nn.MSELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)

    def train(encoder, decoder, data, edge_index_list, criterion,
              optimizer, L2lambda_reg, cluster_centers, use_clustering, gamma, cluster_label, current_epoch):
        encoder.train()
        decoder.train()
        optimizer.zero_grad()

        #random_indices = torch.randperm(len(cluster_label))[64]
        bottle_vec = encoder(data, edge_index_list)
        #bottle_vec_batch = bottle_vec[random_indices]
        #cluster_label_batch = cluster_label[random_indices]

        #cluster_loss = F.smooth_l1_loss(bottle_vec_batch, cluster_label_batch)
        kl_loss = torch.tensor(0.0)
        if use_clustering and cluster_centers is not None:
            kl_loss = self_supervised_clustering(bottle_vec, cluster_centers, gamma)

        Pre_feature = decoder(bottle_vec)

        graph_loss = criterion(Pre_feature, data)

        reg_loss = 0
        for param in encoder.parameters():
            reg_loss += param.norm(2)
        for param in decoder.parameters():
            reg_loss += param.norm(2)
        graph_loss += L2lambda_reg * reg_loss
        graph_loss += kl_loss

        if current_epoch < 20:
            loss = graph_loss
        else:
            loss = graph_loss

        loss.backward()
        optimizer.step()

        batch_size = 50
        total_accuracy = 0
        total_batches = 0

        return loss.item(), kl_loss.item()

    write_log(hyperparameters)

    num_epochs = 10000

    L2lambda_reg = 0.001
    cluster_centers = None
    if use_self_supervised_clustering:
        cluster_centers_df = pd.read_csv('data_compare/original_cluster_centers.csv')
        cluster_centers = torch.tensor(cluster_centers_df.values).float().to(device)
        cluster_centers.requires_grad = True
        optimizer.add_param_group({'params': cluster_centers})

    def load_region_data(region, base_dir, device, load_cluster_label):
        region_dir = os.path.join(base_dir, region)

        data_path = os.path.join(region_dir, f'{region}_data.pt')
        data = torch.load(data_path).x
        


        if data.shape[0] <= 100:
            return None, None, None, None

        data = torch.tensor(data)

        edge_index_path = os.path.join(region_dir, f'{region}_edge.pt')
        edge_index_list = torch.load(edge_index_path)


        cluster_label = None
        if load_cluster_label:
            cluster_label_path = os.path.join(region_dir, f'{region}_umap_label.pt')
            cluster_label = torch.load(cluster_label_path).to(device).float()

        return data, edge_index_list, cluster_label

    data_filename = 'data/data/regions/region_class.csv'
    regions = pd.read_csv(data_filename, usecols=[0]).iloc[:, 0]
    base_dir = "data/graph"
    min_loss_per_region = {}
    saved_model_per_region = {}

    for epoch in range(num_epochs):
        if epoch % 20 == 0:
            current_region = regions[epoch // 10 % len(regions)]
            write_log(f"changing train data to {current_region}...")

            data, edge_index_list, cluster_label = load_region_data(
                current_region, base_dir, device, load_cluster_label=False
            )
            if data is None:
                write_log(f"Skipping {current_region} due to insufficient data.")
                continue

            data = data.to(device)
            edge_index_list = [edge_index.to(device) for edge_index in edge_index_list]

        loss, kl_loss = train(Encoder, Decoder, data, edge_index_list, criterion,
                                        optimizer, L2lambda_reg, cluster_centers,
                                        use_clustering=use_self_supervised_clustering, gamma=0.5, cluster_label=cluster_label,
                                        current_epoch=epoch)

        scheduler.step()
        write_log(f"Epoch: {epoch + 1}, Loss: {loss:.4f}")

        if current_region not in min_loss_per_region:
            min_loss_per_region[current_region] = float('inf')

        if loss < min_loss_per_region[current_region]:
            min_loss_per_region[current_region] = loss

            if current_region in saved_model_per_region:
                os.remove(saved_model_per_region[current_region])

            model_filename = f"{current_region}_{loss:.4f}.pt"
            saved_model_path = os.path.join("run/model", model_filename)
            torch.save({'encoder_state_dict': Encoder.state_dict(),
                        'decoder_state_dict': Decoder.state_dict()}, saved_model_path)
            write_log(f"new loss model:{model_filename}")

            saved_model_per_region[current_region] = saved_model_path

        if use_self_supervised_clustering:
            if not os.path.exists('run/cluster_centers'):
                os.makedirs('run/cluster_centers')
            if kl_loss < best_kl_loss:
                best_kl_loss = kl_loss
                cluster_center_save_path = f"cluster_centers/epoch_{epoch + 1}_Loss_{best_loss:.4f}.csv"
                cluster_centers_df = pd.DataFrame(cluster_centers.cpu().detach().numpy())
                cluster_centers_df.to_csv(cluster_center_save_path, index=False)
                write_log(f"Cluster centers saved: {cluster_center_save_path}")

if __name__ == '__main__':
    main()
