import argparse
import time

import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
from torchvision import models

from .dataset import Dataset
from .data_utils import load_train_df, TRAIN_ROOT, CLASSES


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='resnet50')
    arg('--device', default='cpu', choices=['cpu', 'cuda', 'tpu'])
    arg('--lr', type=float, default=0.001)
    arg('--batch-size', type=int, default=16)
    arg('--workers', type=int, default=1)
    arg('--report-each', type=int, default=100)
    arg('--tpu-metrics', action='store_true')
    args = parser.parse_args()
    if args.device == 'tpu':
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_worker, args=(args,))
    else:
        _worker(0, args)


def _worker(worker_index, args):
    if args.device == 'cuda':
        torch.backends.cudnn.benchmark = True
    on_tpu = args.device == 'tpu'
    if on_tpu:
        print(f'started worker {worker_index}')
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(args.device)
    print(f'using device {device}')

    train_df = load_train_df()
    # limit to the part which is already loaded
    present_ids = {p.stem for p in TRAIN_ROOT.glob('*.dcm')}
    train_df = train_df[train_df['Image'].isin(present_ids)]
    dataset = Dataset(df=train_df, root=TRAIN_ROOT)
    print(f'{len(dataset):,} items in train')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )

    model = getattr(models, args.model)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    step = 0

    def train():
        nonlocal step

        model.train()
        if on_tpu:
            import torch_xla.distributed.parallel_loader as pl
            para_loader = pl.ParallelLoader(loader, [device])
            # would need enumerate in xla>0.5
            i_loader = para_loader.per_device_loader(device)
        else:
            i_loader = enumerate(loader)

        total_times, data_times, compute_times = [], [], []
        data_t0 = t0 = time.perf_counter()
        for i, (x, y) in i_loader:
            _t0 = time.perf_counter()
            data_times.append(_t0 - data_t0)
            total_times.append(_t0 - t0)
            t0 = _t0

            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            if on_tpu:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            data_t0 = time.perf_counter()
            compute_times.append(data_t0 - t0)
            if i % args.report_each == 0:
                print(
                    f'[worker {worker_index}]',
                    f'step={step:,}',
                    f'loss={loss:.4f}',
                    f'data_time={np.mean(data_times):.3f}',
                    f'compute_time={np.mean(compute_times):.3f}',
                    f'total_time={np.mean(total_times):.3f}',
                )
                data_times.clear()
                compute_times.clear()
                total_times.clear()
            step += 1

    try:
        train()
    except KeyboardInterrupt:
        print('Ctrl+C pressed, interrupting...')
    finally:
        if on_tpu and args.tpu_metrics:
            import torch_xla.debug.metrics as met
            print('\nTPU metrics report:')
            print(met.metrics_report())


if __name__ == '__main__':
    main()
