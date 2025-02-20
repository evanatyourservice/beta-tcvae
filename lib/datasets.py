import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os

import lib.datasets as dset


class Shapes(Dataset):
    def __init__(self, dataset_zip=None):
        loc = 'data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if dataset_zip is None:
            if not os.path.exists(loc):
                raise FileNotFoundError(f"Dataset not found at {loc}. Please download dSprites dataset.")
            with np.load(loc, encoding='latin1') as dataset_zip:
                self.imgs = torch.from_numpy(dataset_zip['imgs']).float()
        else:
            self.imgs = torch.from_numpy(dataset_zip['imgs']).float()

    def __len__(self):
        return self.imgs.size(0)

    def __getitem__(self, index):
        x = self.imgs[index].view(1, 64, 64)
        return x  # Return CPU tensor, will be moved to CUDA by DataLoader


class BaseDataset(Dataset):
    def __init__(self, loc):
        if not os.path.exists(loc):
            raise FileNotFoundError(f"Dataset not found at {loc}")
        self.dataset = torch.load(loc).float().div(255).view(-1, 1, 64, 64)

    def __len__(self):
        return self.dataset.size(0)

    def __getitem__(self, index):
        return self.dataset[index]  # Return CPU tensor, will be moved to CUDA by DataLoader


class Faces(BaseDataset):
    LOC = 'data/basel_face_renders.pth'

    def __init__(self):
        super(Faces, self).__init__(self.LOC)


def setup_data_loaders(args):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=3,
        generator=torch.Generator()
    )
    return train_loader
