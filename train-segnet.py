import collections.abc
import argparse
import logging
import sys
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from utils.edge_dist import dist_loss
from evaluate import evaluate
from segnet import SegNet

dir_img = Path('./slicedata/imgs/')
dir_mask = Path('./slicedata/masks/')
dir_valimg = Path('./sliceval/imgs/')
dir_valmask = Path('./sliceval/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_net(net,
              device,
              epochs: int = 25,
              batch_size: int = 4,
              learning_rate: float = 1e-4,
              save_checkpoint: bool = True,
              amp: bool = False,
              dim: int = 3,
              dist_epoch: int = 15):
    # 1. Create dataset
    train_set = BasicDataset(dir_img, dir_mask)
    val_set = BasicDataset(dir_valimg, dir_valmask)

    # 2. Train / validation partitions
    n_val = len(val_set)
    n_train = len(train_set)

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    criterion = nn.BCELoss()  # changed here for 1 class output
    # smooth = nn.MSELoss()

    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']

                true_masks = batch['mask']

                images = images.to(device=device, dtype=torch.float32)

                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                masks_pred = net(images)
                
                loss = criterion(masks_pred, true_masks) + dice_loss(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()

                pbar.set_postfix(**{'Original loss': loss.item()}) # , 'Smooth loss': loss2.item()})

                # Evaluation round
                division_step = (n_train // (2 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(dim, epoch+1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--slicedim', '-sd', dest='slice_dim', metavar='SD', type=int, default=3, help='dimension number of slices')
    parser.add_argument('--distepoch', '-de', dest='dist_epoch', metavar='DE', type=int, default=15, help='time to start distance loss')


    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    net = SegNet(input_nbr=1,label_nbr=1)

    net.to(device=device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  amp=args.amp,
                  dim=args.slice_dim,
                  dist_epoch=args.dist_epoch)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './checkpoints/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
