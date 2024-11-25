from torch.utils.data import Dataset
import torch
import numpy as np


class BlockDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        data = np.load(root)
        self.images = torch.from_numpy(data['images'])
        self.vectors = torch.from_numpy(data['vectors'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.vectors[idx]


if __name__ == '__main__':
    from gen_data import vector2language
    ds = BlockDataset('./data/8blocks-1000.npz')
    print(ds[0][0], ds[0][1], vector2language(ds[0][1]))
