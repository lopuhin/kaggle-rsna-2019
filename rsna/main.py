import argparse
import time

import numpy as np
import torch
from torch import nn, optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from .dataset import Dataset
from .data_utils import load_train_df, TRAIN_ROOT, CLASSES, train_valid_split


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--model', default='resnet50')
    arg('--device', default='tpu', choices=['cpu', 'cuda', 'tpu'])
    arg('--lr', type=float, default=0.001)
    arg('--batch-size', type=int, default=24)
    arg('--workers', type=int, default=2)
    arg('--report-each', type=int, default=100)
    arg('--tpu-metrics', action='store_true')
    arg('--fold', type=int, default=0)
    arg('--n-folds', type=int, default=5)
    arg('--epochs', type=int, default=10)
    arg('--epoch-steps', type=int)
    arg('--valid-steps', type=int)
    arg('--comment', default='', help='postfix for tensorboard run folder')
    args = parser.parse_args()

    train_df = load_train_df()  # do initial load in one process only
    if args.device == 'tpu':
        import torch_xla.distributed.xla_multiprocessing as xmp
        xmp.spawn(_worker, args=(args, train_df))
    else:
        _worker(0, args, train_df)


def _worker(worker_index, args, train_df):
    writer = SummaryWriter(comment=args.comment, flush_secs=10)
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

    # limit to the part which is already loaded - FIXME
    present_ids = {p.stem for p in TRAIN_ROOT.glob('*.dcm')}
    train_df = train_df[train_df['Image'].isin(present_ids)]
    train_df, valid_df = train_valid_split(
        train_df, fold=args.fold, n_folds=args.n_folds)
    train_dataset = Dataset(df=train_df, root=TRAIN_ROOT)
    valid_dataset = Dataset(df=valid_df, root=TRAIN_ROOT)
    print(f'{len(train_dataset):,} items in train, '
          f'{len(valid_dataset):,} items in valid')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        drop_last=True,
    )
    epoch_steps = args.epoch_steps or len(train_loader)  # FIXME TPU?
    print(f'epoch size {epoch_steps:,}')
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
    )
    valid_steps = args.valid_steps or len(valid_loader)

    model = getattr(models, args.model)(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    step = 0

    def train():
        nonlocal step
        model.train()
        total_times, data_times, compute_times = [], [], []
        data_t0 = t0 = time.perf_counter()
        for i, (x, y) in enumerate(
                _iter_loader(train_loader, device, on_tpu=on_tpu)):
            _t0 = time.perf_counter()
            data_times.append(_t0 - data_t0)
            total_times.append(_t0 - t0)
            t0 = _t0

            if not on_tpu:
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
            if step % args.report_each == 0:
                writer.add_scalar('loss/train', loss, step)
                print(
                    f'[worker {worker_index}]',
                    f'training step {step:,}/{epoch_steps * args.epochs:,}',
                    f'loss={loss:.4f}',
                    f'data_time={np.mean(data_times):.3f}',
                    f'compute_time={np.mean(compute_times):.3f}',
                    f'total_time={np.mean(total_times):.3f}',
                )
                data_times.clear()
                compute_times.clear()
                total_times.clear()
            step += 1
            if i >= epoch_steps - 1:
                break

    def evaluate():
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (x, y) in enumerate(
                    _iter_loader(valid_loader, device, on_tpu=on_tpu)):
                if not on_tpu:
                    x = x.to(device)
                    y = y.to(device)
                y_pred = model(x)
                losses.append(float(criterion(y_pred, y)))  # FIXME TPU speed
                if i % args.report_each == 0:
                    print(
                        f'[worker {worker_index}]',
                        f'evaluation step {i:,}/{valid_steps:,}',
                        f'loss={np.mean(losses):.4f}')
                if i >= valid_steps - 1:
                    break
            loss = np.mean(losses)
            writer.add_scalar('loss/valid', loss, step)
            print(f'loss at step {step}: {loss:.4f}')

    try:
        for _ in range(args.epochs):
            train()
            evaluate()
    except KeyboardInterrupt:
        print('Ctrl+C pressed, interrupting...')
    finally:
        if on_tpu and args.tpu_metrics:
            import torch_xla.debug.metrics as met
            print('\nTPU metrics report:')
            print(met.metrics_report())


def _iter_loader(loader, device, on_tpu):
    if on_tpu:
        import torch_xla.distributed.parallel_loader as pl
        para_loader = pl.ParallelLoader(loader, [device])
        # would not be needed in xla>0.5
        return (xy for _, xy in para_loader.per_device_loader(device))
    else:
        return iter(loader)


if __name__ == '__main__':
    main()
