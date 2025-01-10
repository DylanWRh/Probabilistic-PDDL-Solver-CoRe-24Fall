import numpy as np
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('..')


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
        data = [self.tasks[index][key]['data'] for key in ['init'] +
                ['step_{}'.format(i + 1) for i in range(self.n_steps)]]
        label = [self.tasks[index][key]['label'] for key in ['init'] +
                 ['step_{}'.format(i + 1) for i in range(self.n_steps)]]
        return torch.from_numpy(np.stack(data)).flatten(-2, -1).float(), torch.from_numpy(np.stack(label)).flatten(-2, -1).float()
