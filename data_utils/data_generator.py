import os
import sys
sys.path.append('..')
from tqdm import tqdm
import random
import argparse
import numpy as np
from environment import BlockStackingEnv

def gen_data(args):
    xx, yy = np.meshgrid(
        np.arange(args.x_min, args.x_max+1),
        np.arange(args.y_min, args.y_max+1),
    )
    x = xx.flatten()
    y = yy.flatten()
    grid = np.stack([x, y], axis=1)
    n = len(grid)
        
    
    env = BlockStackingEnv(args.n_blocks)
    
    n_samples = args.n_samples_train + args.n_samples_val + args.n_samples_test 
    
    datas = []
    labels = []
    
    for _ in tqdm(range(n_samples)):
        list_state = [[i] for i in range(1, args.n_blocks+1)]
        num_step = random.randint(1, 1000)
        for t in range(num_step):
            i, j = np.random.choice(args.n_blocks, 2, replace=False)
            if list_state[i]:
                obj_i = list_state[i].pop(-1)
                list_state[j].append(obj_i)
        
        coords = np.random.choice(n, args.n_blocks, replace=False)
        block_coords = grid[coords]
        env.set_coords(block_coords)
        env.set_list_state(list_state)
        
        data = env.get_coords_3d()
        label = env.vector_state
        
        datas.append(data)
        labels.append(label)
        
    datas = np.stack(datas, axis=0)
    labels = np.stack(labels, axis=0)
    
    os.makedirs(args.data_path, exist_ok=True)
    train_data = datas[:args.n_samples_train]
    val_data = datas[args.n_samples_train:args.n_samples_train + args.n_samples_val]
    test_data = datas[args.n_samples_train + args.n_samples_val:]
    
    train_labels = labels[:args.n_samples_train]
    val_labels = labels[args.n_samples_train:args.n_samples_train + args.n_samples_val]
    test_labels = labels[args.n_samples_train + args.n_samples_val:]
    
    np.savez(
        os.path.join(args.data_path, f'{args.n_blocks}blocks-{args.n_samples_train}_train.npz'), 
        n_blocks=args.n_blocks,
        coords=train_data, 
        labels=train_labels)
    np.savez(
        os.path.join(args.data_path, f'{args.n_blocks}blocks-{args.n_samples_val}_val.npz'), 
        n_blocks=args.n_blocks,
        coords=val_data, 
        labels=val_labels)
    np.savez(
        os.path.join(args.data_path, f'{args.n_blocks}blocks-{args.n_samples_test}_test.npz'), 
        n_blocks=args.n_blocks,        
        coords=test_data, 
        labels=test_labels)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/states')
    parser.add_argument('--n_samples_train', type=int, default=3000)
    parser.add_argument('--n_samples_val', type=int, default=500)
    parser.add_argument('--n_samples_test', type=int, default=500)
    parser.add_argument('--x_min', type=int, default=-3)
    parser.add_argument('--x_max', type=int, default=3)
    parser.add_argument('--y_min', type=int, default=-3)
    parser.add_argument('--y_max', type=int, default=3)
    parser.add_argument('--n_blocks', type=int, default=8)
    args = parser.parse_args()
    
    gen_data(args)
        