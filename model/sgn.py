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
        
        self.object_encoders = nn.ModuleList()
        
        for i in range(n_blocks):
            self.object_encoders.append(ObjectEncoder(n_blocks * 3, hidden_dim, hidden_dim, depth))
        
        self.clear_encoder = PredicateEncoder(hidden_dim, hidden_dim, 1, depth)
        self.ontable_encoder = PredicateEncoder(hidden_dim, hidden_dim, 1, depth)
        self.AonB_encoder = PredicateEncoder(hidden_dim * 2, hidden_dim, 1, depth)

    
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
                        torch.cat([object_encodings[i], object_encodings[j]], dim=-1)))
            predicate_encodings.append(self.clear_encoder(object_encodings[i]))
            predicate_encodings.append(self.ontable_encoder(object_encodings[i]))
        
        results = torch.cat(predicate_encodings, dim=-1)
        return results
        

if __name__ == '__main__':
    model = BlockStackingSGN(8, 64, 6)
    x = torch.randn(32, 8*3)
    y = model(x)
    print(y.shape)