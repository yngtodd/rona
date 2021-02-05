import random

from PIL import Image
from typing import List
from pathlib import Path
from collections import namedtuple
from torch.utils.data import Dataset


MetaData = namedtuple('MetaData', ['path', 'label'])


class RonaData(Dataset):
    """Coronavirus CT data

    Loads the raw images from the UTKML competition.
    Note that the images have varying sizes, and so
    we must use `torchvision.transforms` to resize
    them, otherwise you will get an error when you
    attempt to stack the tensors in a batch.

    Examples::
        transforms = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        data = RonaData(<path>, transforms)
    """
    mean = [0.6405]
    std  = [0.2562]

    def __init__(self, root: str, split: str="train", transform=None):
        self.root = root
        self.transform = transform
        self.meta = self.load_meta(split)

    def __repr__(self):
        return f"RonaData(root={self.root})"

    def __len__(self):
        return len(self.meta)

    def read_img(self, path):
        """Read in an image"""
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert("RGB")

    def load_meta(self, split):
        assert split in ["train", "test"]

        if split == "test":
            return self.load_test_meta()
        else:
            return self.load_train_meta()

    def load_train_meta(self) -> List[MetaData]:
        """Load metadata

        We want to keep track of the paths for each
        of our samples along with their labels. This
        is stored as a list of `MetaData`, a named tuple
        that stores each sample's path and its label.
        """
        root = Path(self.root)
        rona = root.joinpath("COVID").glob("**/*.png")
        safe = root.joinpath("non-COVID").glob("**/*.png")

        samples = []
        for path in rona:
            samples.append(
                MetaData(path, label=1.0)
            )

        for path in safe:
            samples.append(
                MetaData(path, label=0.0)
            )

        random.shuffle(samples)

        return samples

    def load_test_meta(self) -> List[MetaData]:
        root = Path(self.root)
        test = root.joinpath("test_data").glob("**/*.png")

        samples = []
        for path in test:
            samples.append(
                MetaData(path, label=None)
            )

        return samples

    def __getitem__(self, idx: int):
        path, target = self.meta[idx]

        sample = self.read_img(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

