import os
import sys
sys.path.append('..')
from tqdm import tqdm
import argparse
import numpy as np
from environment import BlockStackingEnv

def gen_data(args):
    assert args.z_min >= 0
    xx, yy, zz = np.meshgrid(
        np.arange(args.x_min, args.x_max+1),
        np.arange(args.y_min, args.y_max+1),
        np.arange(args.z_min, args.z_max+1)
    )
    x = xx.flatten()
    y = yy.flatten()
    z = zz.flatten()
    grid = np.stack([x, y, z], axis=1)
    n = len(grid)
    
    coords = []
    labels = []
    env = BlockStackingEnv(args.n_blocks)
    
    n_samples = args.n_samples_train + args.n_samples_val + args.n_samples_test 
    
    for i in tqdm(range(n_samples)):
        blocks = np.random.choice(n, args.n_blocks, replace=False)
        block_coords = grid[blocks]   # (N, 3)
        env.set_coords_state(block_coords)
        label = env.vector_state        # (N, N+2)
        
        coords.append(block_coords)
        labels.append(label)
        
    coords = np.stack(coords, axis=0)
    labels = np.stack(labels, axis=0)
    
    os.makedirs(args.data_path, exist_ok=True)
    
    train_coords = coords[:args.n_samples_train]
    val_coords = coords[args.n_samples_train:args.n_samples_train + args.n_samples_val]
    test_coords = coords[args.n_samples_train + args.n_samples_val:]

    train_labels = labels[:args.n_samples_train]
    val_labels = labels[args.n_samples_train:args.n_samples_train + args.n_samples_val]
    test_labels = labels[args.n_samples_train + args.n_samples_val:]

    np.savez(
        os.path.join(args.data_path, f'{args.n_blocks}blocks-{args.n_samples_train}_train.npz'), 
        n_blocks=args.n_blocks,
        coords=train_coords, 
        labels=train_labels)
    np.savez(
        os.path.join(args.data_path, f'{args.n_blocks}blocks-{args.n_samples_val}_val.npz'), 
        n_blocks=args.n_blocks,
        coords=val_coords, 
        labels=val_labels)
    np.savez(
        os.path.join(args.data_path, f'{args.n_blocks}blocks-{args.n_samples_test}_test.npz'), 
        n_blocks=args.n_blocks,        
        coords=test_coords, 
        labels=test_labels)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--n_samples_train', type=int, default=3000)
    parser.add_argument('--n_samples_val', type=int, default=500)
    parser.add_argument('--n_samples_test', type=int, default=500)
    parser.add_argument('--x_min', type=int, default=-1)
    parser.add_argument('--x_max', type=int, default=1)
    parser.add_argument('--y_min', type=int, default=-1)
    parser.add_argument('--y_max', type=int, default=1)
    parser.add_argument('--z_min', type=int, default=0)
    parser.add_argument('--z_max', type=int, default=3)
    parser.add_argument('--n_blocks', type=int, default=8)
    args = parser.parse_args()
    
    gen_data(args)
        