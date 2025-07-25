import torch
import torch.nn as nn
from fusion_layer import hyper_fusion_Layer


class Net(nn.Module):
    def __init__(self,d_model):
        super(Net, self).__init__()
        self.R = 8
        self.s_dim = 5
        self.f_dim = 2*d_model
        self.n_output = 2*d_model
        self.mlp_signal = nn.Sequential(
            nn.Linear(self.f_dim, self.f_dim),
            nn.ReLU(),
            nn.Linear(self.f_dim, self.s_dim)
        )
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.layer_test = hyper_fusion_Layer(self.R, self.s_dim, self.f_dim, self.n_output)
        self.activation = nn.LeakyReLU()
        self.layer_norm = nn.LayerNorm(self.n_output)

    def forward(self, merge_embedded, position_embedded, batch_size, seq_length, batch_num):
        feature = torch.cat([merge_embedded, position_embedded], dim=2)
        signal = self.mlp_signal(feature)
        x = self.layer_test(signal, feature, batch_size, seq_length)
        x = self.activation(x)
        x = self.layer_norm(x)
        return x