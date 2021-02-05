import torch.optim as optim
import torchvision.models as models

from rona.data import RonaData
from torchvision import transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='MTCNN Baseline')
    parser.add_argument('--dataroot', type=str,
                        help='Root path to the data')
    parser.add_argument('--savepath', type=str,
                        help='path to the save loss curves')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--device', type=str, default="cuda",
                        help='cpu or cuda')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Adam learning rate')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Adam learning rate')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='Adam epsilon')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(ARGS.device)

    data_transforms = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    data = RonaData(args.dataroot, data_transforms)
    data_loader = DataLoader(data, args.batch_size)


    optimizer = optim.Adam(
        model.parameters(), lr=ARGS.lr, eps=ARGS.eps
    )


if __name__=="__main__":
    main()
