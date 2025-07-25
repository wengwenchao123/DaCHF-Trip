import torch
import torch.nn as nn


class hyper_fusion_Layer(nn.Module):
    def __init__(self, R, s_dim, in_dim, out_dim):
        super(hyper_fusion_Layer, self).__init__()
        self.R = R
        self.s_dim = s_dim
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.A = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.in_dim, self.R)))
        self.B = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.R, self.out_dim)))
        self.C = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.s_dim, self.R)))
        self.M_1 = nn.Parameter(torch.nn.init.xavier_uniform_(torch.Tensor(self.in_dim, self.out_dim)))

    def forward(self, signal, feature, batch_size, seq_length):
        AB_test = torch.matmul(self.A, self.B).unsqueeze(0)
        ABC_test = torch.einsum("sd,dnm->snm",self.C, AB_test)
        T_1_test = ABC_test
        w_1_test = torch.einsum("bls,snm->blnm", signal, T_1_test)
        x = torch.einsum("bld,bldh->blh",feature, (w_1_test + self.M_1))
        return x