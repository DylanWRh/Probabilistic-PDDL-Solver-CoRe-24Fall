import torch
import torch.nn as nn
import torch.nn.functional as F


class NaiveMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth, use_sigmoid=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.layers = nn.Sequential(
            *(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU()] +
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (depth - 1) +
                [nn.Linear(hidden_dim, output_dim), nn.Sigmoid() if use_sigmoid else nn.ReLU()]
            )
        )
        # self.layers.append(nn.Linear(input_dim, hidden_dim))
        # for i in range(depth-1):
        #     self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        # self.layers.append(nn.Linear(hidden_dim, output_dim))

        # self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # for i in range(self.depth):
        #     x = self.activation(self.layers[i](x))
        #     x = self.dropout(x)
        # x = self.layers[self.depth](x)
        return self.layers(x)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=24, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * torch.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                       torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                       torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(
            torch.cat([self.embed(input, self.basis), input], dim=2))  # B x N x C
        return embed


class PosEmbdMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, n_layers):
        super().__init__()
        self.point_embd = PointEmbed(dim=embed_dim)
        self.layers = nn.Sequential(
            *(
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] +
                [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 1) +
                [nn.Linear(hidden_dim, output_dim), nn.Sigmoid()]
            )
        )

    def forward(self, x):
        B, N = x.shape
        x = self.point_embd(x.reshape(B, -1, 3)).reshape(B, -1)
        return self.layers(x)
