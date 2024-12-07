from model.sgn import BlockStackingSGN
from data_utils.dataset import BlockStackingDataset

import numpy as np
import torch
import argparse
from environment import BlockStackingEnv
from planner import BlockStackingPlanner

def main():
    parser = argparse.ArgumentParser()
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--checkpoint', type=str, 
                        default='checkpoints/sgn-20241208-022616/model_010.pth')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='./data/8blocks-500_test.npz')
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    ds = BlockStackingDataset(args.data)
    model = BlockStackingSGN(ds.n_blocks, hidden_dim=args.hidden_dim, depth=args.depth)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    
    t = np.random.randint(0, len(ds))
    coords1, labels1 = ds[t]
    coords2, labels2 = ds[493]
    labels1 = labels1.view(ds.n_blocks, -1).cpu().numpy()
    labels2 = labels2.view(ds.n_blocks, -1).cpu().numpy()
    with torch.no_grad():
        logits1 = model(coords1)
        logits2 = model(coords2)
    probs1 = torch.sigmoid(logits1)
    probs2 = torch.sigmoid(logits2)
    probs1 = probs1.view(ds.n_blocks, -1).cpu().numpy()
    probs2 = probs2.view(ds.n_blocks, -1).cpu().numpy()
        
    bsenv = BlockStackingEnv(ds.n_blocks)
    bsplanner = BlockStackingPlanner(bsenv, probs1, probs2)
    
    bsplanner.run()
    

if __name__ == '__main__':
    main()
    