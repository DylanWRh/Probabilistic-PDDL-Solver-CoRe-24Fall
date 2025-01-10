from .encoder import ObjectEncoder, PredicateEncoder

import torch
import torch.nn as nn


class BlockStackingSGN(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        hidden_dim: int,    # input and output dim determined by n_blocks
        depth: int
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.object_encoders = nn.ModuleList(
            [ObjectEncoder(3 * n_blocks, hidden_dim, hidden_dim, depth) for _ in range(self.n_blocks)]
        )
        self.clear_encoder = PredicateEncoder(hidden_dim, hidden_dim, 1, depth, use_sigmoid=True)
        self.ontable_encoder = PredicateEncoder(hidden_dim, hidden_dim, 1, depth, use_sigmoid=True)
        self.AonB_encoder = PredicateEncoder(hidden_dim * 2, hidden_dim, 1, depth, use_sigmoid=True)

    def forward(self, x):
        ''' x: (B, N*3)'''
        object_encodings = []
        predicate_encodings = []

        for i in range(self.n_blocks):
            object_encodings.append(self.object_encoders[i](x))

        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                predicate_encodings.append(
                    self.AonB_encoder(
                        torch.cat(
                            [
                                object_encodings[i],
                                object_encodings[j]
                            ],
                            dim=-1
                        )
                    )
                )
            predicate_encodings.append(self.clear_encoder(object_encodings[i]))
            predicate_encodings.append(self.ontable_encoder(object_encodings[i]))

        results = torch.cat(predicate_encodings, dim=-1)
        return results
