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


class BlockStackingTaskDataset:
    def __init__(self, root):
        super().__init__()
        data = np.load(root, allow_pickle=True)
        self.n_blocks = data['n_blocks']
        self.tasks = data['tasks']
    
    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, idx):
        return self.tasks[idx]
    

class BlockStackingDemonstration(Dataset):
    def __init__(self, root, n_samples):
        super().__init__()
        data = np.load(root, allow_pickle=True)
        self.n_blocks = data['n_blocks']
        tasks = data['tasks']
        self.tasks = tasks[np.random.choice(len(tasks), n_samples, False)]
        self.n_steps = data['n_steps']


    def __len__(self):
        return len(self.tasks)
    
    def __getitem__(self, index):
        data = [self.tasks[index][key]['data'] for key in ['init'] + ['step_{}'.format(i + 1) for i in range(self.n_steps)]]
        # image = [self.tasks[index][key]['image'] for key in ['init'] + ['step_{}'.format(i + 1) for i in range(self.n_steps)]]
        label = [self.tasks[index][key]['label'] for key in ['init'] + ['step_{}'.format(i + 1) for i in range(self.n_steps)]]
        return torch.from_numpy(np.stack(data)).flatten(-2, -1).float(), torch.from_numpy(np.stack(label)).flatten(-2, -1).float()


if __name__ == '__main__':
    ds = BlockStackingDataset('../data/states/8blocks-3000_train.npz')
    env = BlockStackingEnv(ds.n_blocks)
    
    idx = np.random.randint(len(ds))
    n_blocks = ds.n_blocks
    coords, labels = ds[idx]
    coords = coords.view(n_blocks, 3)
    labels = labels.view(n_blocks, -1)

    print(coords.shape)
    print(labels.shape)
    # print(ds[0][0], ds[0][1], vector2language(ds[0][1]))
