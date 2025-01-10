import torch.nn as nn


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

    def forward(self, x):
        return self.layers(x)
