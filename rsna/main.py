import argparse
from typing import Tuple

import torch
from torch import nn, optim
import torch.utils.data
from torchvision import models
import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, size: Tuple[int, int], n: int):
        self.size = size
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        w, h = self.size
        return torch.randn(3, h, w, dtype=torch.float32), 0


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='resnet50')
    arg('--device', default='cpu', choices=['cpu', 'cuda', 'tpu'])
    arg('--lr', type=float, default=0.01)
    arg('--batch-size', type=int, default=16)
    arg('--image-size', type=lambda x: tuple(x.split('x')), default=(512, 512))
    arg('--epoch-size', type=int, default=10000)
    arg('--workers', type=int, default=2)
    arg('--report-each', type=int, default=10)
    args = parser.parse_args()

    on_tpu = args.device == 'tpu'
    if on_tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(args.device)
    print(f'using device {device}')

    dataset = Dataset(size=args.image_size, n=args.epoch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    model = getattr(models, args.model)()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    model.train()
    pbar = tqdm.tqdm(loader)
    for i, (x, y) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if on_tpu:
            xm.optimizer_step(optimizer, barrier=True)
        if i % args.report_each == 0:
            pbar.set_postfix(loss=f'{loss:.4f}')


if __name__ == '__main__':
    main()
