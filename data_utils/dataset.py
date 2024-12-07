import sys
sys.path.append('..')
from environment import BlockStackingEnv

from torch.utils.data import Dataset
import torch
import numpy as np


class BlockStackingDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        data = np.load(root)
        self.n_blocks = data['n_blocks']
        self.coords = torch.from_numpy(data['coords']).float()
        self.labels = torch.from_numpy(data['labels']).float()

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords[idx].flatten(), self.labels[idx].flatten()


if __name__ == '__main__':
    ds = BlockStackingDataset('../data/8blocks-3000_train.npz')
    env = BlockStackingEnv(ds.n_blocks)
    
    idx = np.random.randint(len(ds))
    n_blocks = ds.n_blocks
    coords, labels = ds[idx]
    coords = coords.view(n_blocks, 3)
    labels = labels.view(n_blocks, -1)
    env.set_coords_state(coords.cpu().numpy())
    print(coords)
    print(labels)
    print(env.vector_state)
    print(env.image_state)
    print(env.get_language_state())
    # print(ds[0][0], ds[0][1], vector2language(ds[0][1]))
