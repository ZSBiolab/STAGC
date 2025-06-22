import torch.nn as nn
from layer import Attn_head
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channel, bottle_size, attn_drop, ffd_drop,
                 hid_units, n_heads, activation=nn.SELU(), residual=True, nn_liner_batch_size=32):
        super(Encoder, self).__init__()

        if len(hid_units) != len(n_heads):
            raise ValueError("The lengths of hid_units and n_heads must be the same.")

        self.bottle_vec = bottle_size
        #self.nb_nodes_list = nb_nodes_list
        self.attn_drop = attn_drop
        self.ffd_drop = ffd_drop
        self.hid_units = hid_units
        self.n_heads = n_heads
        self.activation = activation
        self.residual = residual
        self.in_channel = in_channel
        self.nn_liner_batch_size= nn_liner_batch_size
        print("in_channel provided to Encoder:", in_channel)

        self.attn_layers = nn.ModuleList()
        in_sz = self.in_channel

        for i in range(len(hid_units)):
            self.attn_layers.append(
                nn.ModuleList([Attn_head(in_channel=in_sz, out_sz=hid_units[i], activation=self.activation,
                                         in_drop=self.ffd_drop,
                                         coef_drop=self.attn_drop, residual=self.residual)
                               for _ in range(n_heads[i])]
                              ))
            in_sz = n_heads[i] * hid_units[i]
        self.final_layers = nn.ModuleList([
            nn.Linear(self.hid_units[-1] * self.n_heads[-1], self.hid_units[-1]),

            nn.SELU(),
            nn.Linear(self.hid_units[-1], 32),  #
            nn.SELU(),
            nn.Linear(32, 128),  
            nn.SELU(),
            nn.Linear(128, bottle_size)  
        ])


    def forward_with_minibatching(self, final_embed, batch_size):
        num_nodes = final_embed.shape[0]
        outputs = []


        for start_idx in range(0, num_nodes, batch_size):
            end_idx = min(start_idx + batch_size, num_nodes)
            mini_batch = final_embed[start_idx:end_idx]
            #print("end_idx:", end_idx)

            mini_out = mini_batch
            for layer in self.final_layers:
                mini_out = layer(mini_out)

            outputs.append(mini_out)
            del mini_out
            torch.cuda.empty_cache()

        outputs_gpu = [output.cuda() for output in outputs]

        logits = torch.cat(outputs_gpu, dim=0)

        del outputs
        torch.cuda.empty_cache()
        return logits

    def forward(self, data, edge_index_list):
        inputs = data
        #print("Checking inputs in module4.py")
        #print(inputs)
        #print(type(inputs))

        embed_list = []

        for edge_index in edge_index_list:
           
            attns = [self.attn_layers[0][j](inputs, edge_index) for j in range(self.n_heads[0])]
            h_1_new = torch.cat(attns, dim=-1)
        
            for i in range(1, len(self.hid_units)):
                if i != 1:  
                    del h_1
                    torch.cuda.empty_cache()

                h_1 = h_1_new
                attns = [self.attn_layers[i][j](h_1, edge_index) for j in range(self.n_heads[i])]
                h_1_new = torch.cat(attns, dim=-1)
                #print("Shape of hide_h_1.shape:", h_1.shape)
            # h_1.unsqueeze(1):[num_nodes,1,hid_units[-1]×n_heads[-1]]
            embed_list.append(h_1_new.unsqueeze(1))
            #[6,num_nodes,1,hid_units[-1]×n_heads[-1]]

        multi_embed = torch.cat(embed_list, dim=1)#[num_nodes,6,hid_units[-1]×n_heads[-1]]

   
        bottle_vec = self.forward_with_minibatching(multi_embed, self.nn_liner_batch_size)
        bottle_vec = bottle_vec.squeeze(1)
        #print("Shape of bottle_vec:", bottle_vec.shape)
        return bottle_vec

class SimpleAttention(nn.Module):
    def __init__(self, in_dim):
        super(SimpleAttention, self).__init__()
        # Key, Query, Value projections
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        return attn_output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.ln1 = nn.LayerNorm(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.ln2 = nn.LayerNorm(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = out.transpose(1, 2)  
        out = self.ln1(out)
        out = out.transpose(1, 2) 
        out = F.relu(out)

        out = self.conv2(out)
        out = out.transpose(1, 2)
        out = self.ln2(out)
        out = out.transpose(1, 2)
        out += self.shortcut(x)
        return F.relu(out)

class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, num_res_blocks=3, num_attn_layers=3):
        super(Decoder, self).__init__()
        
        # ResNet layers
        self.res_layers = nn.Sequential(
            *[ResidualBlock(in_channel if i == 0 else out_channel, out_channel) for i in range(num_res_blocks)]
        )

        # Custom attention layers
        self.attn_layers = nn.ModuleList([SimpleAttention(out_channel) for _ in range(num_attn_layers)])

    def forward(self, x):
       
        if x.dim() == 2:
            x = x.unsqueeze(1) 
        # Pass through ResNet layers
        x = x.transpose(1, 2)  # Transpose for Conv1d compatibility [batch_size, channels, seq_len]
        x = self.res_layers(x)
        x = x.transpose(1, 2)  # Transpose back to [batch_size, seq_len, channels]

        # Pass through attention layers
        for attn_layer in self.attn_layers:
            x = attn_layer(x)

        # Apply tanh activation
        return torch.tanh(x)
        #return x