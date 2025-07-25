import torch
import torch.nn.functional as F


class Contrast(torch.nn.Module):
    def __init__(self, num_hidden: int, tau: float = 0.7, lambda_sym: float = 0.1):
        super(Contrast, self).__init__()
        self.tau: float = tau
        self.lambda_sym: float = lambda_sym  # 对称性损失的权重

        self.mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )
        self.mlp2 = torch.nn.Sequential(
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden, num_hidden, bias=True),
        )

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def self_sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return (z1 * z2).sum(1)

    def symmetry_loss(self, h1: torch.Tensor, h2: torch.Tensor):
        # 计算对称性损失
        return 1 - F.cosine_similarity(h1, h2, dim=1).mean()

    def loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.self_sim(z1, z2))
        rand_item = torch.randperm(z1.shape[0])
        neg_sim = f(self.self_sim(z1, z2[rand_item])) + f(self.self_sim(z2, z1[rand_item]))
        # neg_sim = f(self.self_sim(z1, z2[rand_item]))
        # return -torch.log(between_sim / (between_sim + neg_sim))
        return -torch.log(2*between_sim / (2*between_sim  + neg_sim))

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        h1 = self.mlp1(z1)
        h2 = self.mlp2(z2)
        loss = self.loss(h1, h2).mean()
        sym_loss = self.symmetry_loss(h1, h2)
        total_loss = loss + self.lambda_sym * sym_loss  # 加权总损失
        return total_loss
        # return loss