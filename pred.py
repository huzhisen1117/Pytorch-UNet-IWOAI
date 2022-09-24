# test for all the test set and then output 3-D results
import argparse
import logging
import sys
import h5py
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
# import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import listdir
from os.path import splitext
from pathlib import Path

from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss
from evaluate import evaluate
from unet import UNet
from segnet import SegNet
from utils.dice_score import dice_coeff
from utils.DFT import DFT


save_path = './results/'

dir_img = './test/test/'
dir_gt = './test/ground-truth/'

dir_model = './checkpoints/checkpoint_epoch20.pth'

def test_img(net, imgdir, gtdir, device):
    
    net.eval()
    
    dice_score = 0
    dice_score_3d = 0
    count = 0

    filelist = [splitext(file)[0] for file in listdir(dir_img) if not file.startswith('.')]

    tmp = torch.zeros(1,384,384,160,len(filelist))
    pred = torch.zeros(1,384,384,160,len(filelist))

    for i in range(len(filelist)):
        img_file = list(Path(dir_img).glob(filelist[i] + '.im'))
        gt_file = list(Path(dir_gt).glob(filelist[i] + '.npy')) 
        
        with h5py.File(img_file[0],'r') as f:
            img = f['data'][:]
        img = img / img.max()
        img_new = img[np.newaxis,np.newaxis,:,:,:]
        gt = np.load(gt_file[0])
        gt_new = gt[np.newaxis,np.newaxis,:,:,:,:]

        image = torch.as_tensor(img_new.copy()).float().contiguous()
        mask_true = torch.as_tensor(gt_new[:,:,:,:,:,2].copy()).float().contiguous()
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)

        with torch.no_grad():
        # predict the mask
            num_slice = 160
            for j in range(num_slice):
                image_slice = image[:,:,:,:,j]
                mask_tmp = net(image_slice)
                tmp[:, :, :, j, i] = mask_tmp
                mask_pred = (mask_tmp>0.5).float() # threshold = 0.5
                pred[:, :, :, j, i] = mask_pred
                dice_score += dice_coeff(mask_pred, mask_true[:,:,:,:,j])
        
        save_name_tmp = save_path + filelist[i] + '.tmp'
        tmp1 = tmp[:,:,:,:,i]
        tmp2 = tmp1.squeeze()
        with h5py.File(save_name_tmp,'w') as h5f:
            h5f.create_dataset('data',data=tmp2)

        save_name_pred = save_path + filelist[i] + '.pred'
        pred1 = pred[:,:,:,:,i]
        pred2 = pred1.squeeze()
        with h5py.File(save_name_pred,'w') as h5f:
            h5f.create_dataset('data',data=pred2)
        pred2 = pred2.to(device=device, dtype=torch.float32)

        dice_score_3d += dice_coeff(pred2, mask_true[0,0,:,:,:])

    num_test = len(filelist) * num_slice

    return dice_score / num_test, dice_score_3d / len(filelist)


if __name__ == '__main__':

    
    net = UNet(n_channels=1, n_classes=1)
    # net = SegNet(input_nbr=1, label_nbr=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net.to(device=device)
    net.load_state_dict(torch.load(dir_model, map_location=device))

    logging.info('Model loaded!')


    test_score, test_score_3d = test_img(net, dir_img, dir_gt, device)
    

    print('Dice score in dim 3 (2D):', str(test_score))


    print('Dice score in dim 3 (3D):', str(test_score_3d))
