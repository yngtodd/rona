import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pytorch_lightning as pl
import torchvision.models as models

from rona.data import RonaData
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def parse_args():
    parser = argparse.ArgumentParser(description='Resnet18')
    parser.add_argument('--dataroot', type=str,
                        help='Root path to the data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    return parser.parse_args()


class Classifier(nn.Module):

    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Resnet(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = Classifier()

    def forward(self, x):
        return self.resnet(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-2)

    def training_step(self, train_batch, batch_idx):
        data, target = train_batch
        logits = self.resnet(data)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        data, target = valid_batch
        logits = self.resnet(data)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), target)
        self.log('valid_loss', loss)


def main():
    args = parse_args()

    data_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6405], std=[0.2562]
        ),
    ])

    dataset = RonaData(args.dataroot, data_transforms)
    num_train = len(dataset) // (1/(3/4))
    num_valid = len(dataset) - num_train

    train_data, valid_data = random_split(
        dataset, [int(num_train), int(num_valid)]
    )

    train_loader = DataLoader(
        train_data, args.batch_size, num_workers=8
    )
    valid_loader = DataLoader(
        valid_data, args.batch_size, num_workers=8
    )

    model = Resnet()

    trainer = pl.Trainer(gpus=[2], limit_train_batches=0.5)
    trainer.fit(model, train_loader, valid_loader)


if __name__=="__main__":
    main()
