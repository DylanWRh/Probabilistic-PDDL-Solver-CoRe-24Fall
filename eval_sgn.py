from engine_sgn import evaluate
from model.sgn import BlockStackingSGN
from data_utils.dataset import BlockStackingDataset

import torch
import argparse

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
    test_loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = BlockStackingSGN(ds.n_blocks, hidden_dim=args.hidden_dim, depth=args.depth)
    model.load_state_dict(torch.load(args.checkpoint))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    evaluate(model, criterion, test_loader, device, 0)
    
if __name__ == '__main__':
    
    main()