import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, depth):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for i in range(depth-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        for i in range(self.depth):
            x = self.activation(self.layers[i](x))
            x = self.dropout(x)
        x = self.layers[self.depth](x)
        return x