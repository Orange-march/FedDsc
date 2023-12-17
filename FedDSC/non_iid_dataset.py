import torch as th
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, client_dataset, train=True, transform=None, ) -> None:
        self.train = train
        self.transform = transform
        self.data = client_dataset[0]
        self.targets = client_dataset[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target
