import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy import ndimage
import h5py
import random


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, tissue: str = 'FC', test: bool = False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        # self.slice_dim = slice_dim
        # self.idsli = []
        self.tissue = tissue
        self.test = test

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]  # string
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        # for i in range(len(self.ids)):
            # if slice_dim == 1 or slice_dim == 2:
            #     for j in range(384):
            #         if slice_dim == 1:               # x in (x,y,z)
            #             self.idsli.append(str(self.ids[i] + '_slicex_' + str(j)))
            #         else:                            # y in (x,y,z)
            #             self.idsli.append(str(self.ids[i] + '_slicey_' + str(j)))
            # elif slice_dim == 3:
            # for j in range(160):                 # z in (x,y,z)
            #         self.idsli.append(str(self.ids[i] + '_slicez_' + str(j)))    

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_ndarray, is_mask):

        if not is_mask:
            # img_ndarray = img_ndarray[np.newaxis,:,:,:]
            # img_ndarray1 = img_ndarray[:,:,0]
            # img_ndarray2 = img_ndarray[:,:,1]
            # img_ndarray3 = img_ndarray[:,:,2]
            img_ndarray = img_ndarray / img_ndarray.max()  # (x, y) and normalization shape(384,384)
            # img_ndarray2 = img_ndarray2 / img_ndarray2.max()
            # img_ndarray3 = img_ndarray3 / img_ndarray3.max()
            
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1)) # (cls, x, y)
            # img_ndarray1 = img_ndarray[:,:,:,0]
            # img_ndarray2 = img_ndarray[:,:,:,1]
            # img_ndarray3 = img_ndarray[:,:,:,2]

        return img_ndarray

    def rotation_crop(self, img, msk):

        img_pil = Image.fromarray(img)
        # img_pil2 = Image.fromarray(img2)
        # img_pil3 = Image.fromarray(img3)
        msk_pil = Image.fromarray(msk)
        # msk_pil2 = Image.fromarray(msk2)
        # msk_pil3 = Image.fromarray(msk3)

        angles = random.randint(-10,10)

        img_r = img_pil.rotate(angles)
        # img_r2 = img_pil2.rotate(angles)
        # img_r3 = img_pil3.rotate(angles)
        msk_r = msk_pil.rotate(angles)
        # msk_r2 = msk_pil2.rotate(angles)
        # msk_r3 = msk_pil3.rotate(angles)
        
        if self.tissue == 'FC':
            x13 = random.randint(0,40)
            x23 = random.randint(0,60)
            img_rc = img_r.crop((x13,x23,x13+300,x23+300))
            # img_rc2 = img_r2.crop((x13,x23,x13+300,x23+300))
            # img_rc3 = img_r3.crop((x13,x23,x13+300,x23+300))
            msk_rc = msk_r.crop((x13,x23,x13+300,x23+300))
            # msk_rc2 = msk_r2.crop((x13,x23,x13+300,x23+300))
            # msk_rc3 = msk_r3.crop((x13,x23,x13+300,x23+300))
        elif self.tissue == 'TC':
            x13 = random.randint(60,80)
            x23 = random.randint(30,80)
            img_rc = img_r.crop((x13,x23,x13+300,x23+300))
            # img_rc2 = img_r2.crop((x13,x23,x13+300,x23+300))
            # img_rc3 = img_r3.crop((x13,x23,x13+300,x23+300))
            msk_rc = msk_r.crop((x13,x23,x13+300,x23+300))
            # msk_rc2 = msk_r2.crop((x13,x23,x13+300,x23+300))
            # msk_rc3 = msk_r3.crop((x13,x23,x13+300,x23+300))
        elif self.tissue == 'PC':
            x13 = random.randint(0,20)
            x23 = random.randint(0,20)
            img_rc = img_r.crop((x13,x23,x13+300,x23+300))
            # img_rc2 = img_r2.crop((x13,x23,x13+300,x23+300))
            # img_rc3 = img_r3.crop((x13,x23,x13+300,x23+300))
            msk_rc = msk_r.crop((x13,x23,x13+300,x23+300))
            # msk_rc2 = msk_r2.crop((x13,x23,x13+300,x23+300))
            # msk_rc3 = msk_r3.crop((x13,x23,x13+300,x23+300))
        elif self.tissue == 'M':
            x13 = random.randint(60,80)
            x23 = random.randint(30,70)
            img_rc = img_r.crop((x13,x23,x13+300,x23+300))
            # img_rc2 = img_r2.crop((x13,x23,x13+300,x23+300))
            # img_rc3 = img_r3.crop((x13,x23,x13+300,x23+300))
            msk_rc = msk_r.crop((x13,x23,x13+300,x23+300))
            # msk_rc2 = msk_r2.crop((x13,x23,x13+300,x23+300))
            # msk_rc3 = msk_r3.crop((x13,x23,x13+300,x23+300))
        img_d = np.array(img_rc)
        # img_d2 = np.array(img_rc2)
        # img_d3 = np.array(img_rc3)
        msk_d = np.array(msk_rc)
        # msk_d2 = np.array(msk_rc2)
        # msk_d3 = np.array(msk_rc3)
        # img_d = img_tmp[np.newaxis,:,:]
        # msk_d = msk_tmp[np.newaxis,:,:]

        return img_d, msk_d

    @classmethod
    def load(cls, filename):
        # ext = splitext(filename)[1]
        with h5py.File(filename, 'r') as hf:
            img = hf['data'][:]
        
        return img

    def __getitem__(self, idx):
        
        # if self.slice_dim == 1 or self.slice_dim == 2: # slice in x or y direction
             # name = self.ids[idx//384] # filename from the list
        # slice in z direction
        name = self.ids[idx]
    
        img_file = list(self.images_dir.glob(name + '.im'))     # filelist
        img = self.load(img_file[0])
        img = self.preprocess(img, is_mask=False)
        
        if not self.test:
            mask_file = list(self.masks_dir.glob(name + '.seg'))
            mask = self.load(mask_file[0])
        else:
            mask_file = list(self.masks_dir.glob(name + '.npy'))
            mask = np.load(mask_file[0], 'r')

        mask_tmp = self.preprocess(mask, is_mask=True)
        mask = np.zeros(img.shape)
        # mask2 = np.zeros(img2.shape)
        # mask3 = np.zeros(img3.shape)

        # for i in range(tmp.shape[0]):
        #     mask = mask + tmp[i,:,:,:]

        if self.tissue == 'FC':
            mask = mask_tmp[0]
            # mask2 = mask_tmp2[0]
            # mask3 = mask_tmp3[0]
        # elif self.tissue == 'MTC':
        #     mask = mask_tmp[1]
        # elif self.tissue == 'LTC':
        #     mask = mask_tmp[2]
        elif self.tissue == 'TC':
            mask = mask_tmp[1] + mask_tmp[2]
            # mask2 = mask_tmp2[1] + mask_tmp2[2]
            # mask3 = mask_tmp3[1] + mask_tmp3[2]
        elif self.tissue == 'PC':
            mask = mask_tmp[3]
            # mask2 = mask_tmp2[3]
            # mask3 = mask_tmp3[3]
        # elif self.tissue == 'LM':
        #     mask = mask_tmp[4]
        # elif self.tissue == 'MM':
        #     mask = mask_tmp[5]
        elif self.tissue == 'M':
            mask = mask_tmp[4] + mask_tmp[5]
            # mask2 = mask_tmp2[4] + mask_tmp2[5]
            # mask3 = mask_tmp3[4] + mask_tmp3[5]

        # if self.slice_dim == 1:
            # img_slice_tmp = img[idx%384,60:300,:]
            # mask_slice_tmp = mask[idx%384,60:300,:]
            # img_slice_tmp = img[idx%384,:,:]
            # mask_slice_tmp = mask[idx%384,:,:]
            # img_d, mask_d = self.rotation_crop(img_slice_tmp, mask_slice_tmp, dim=1)   # randomly rotate the slice with (-10,10)
            # img_d = img_r[:,60:300,:]
            # mask_d = mask_r[:,60:300,:]
            # blacks = np.zeros((240,40))
            # img_slice = np.concatenate((blacks, img_slice_tmp, blacks), axis=1)
            # mask_slice = np.concatenate((blacks, mask_slice_tmp, blacks), axis=1)
        # elif self.slice_dim == 2:
            # img_slice_tmp = img[40:280,idx%384,:]
            # mask_slice_tmp = mask[40:280,idx%384,:]
            # img_slice_tmp = img[:,idx%384,:]
            # mask_slice_tmp = mask[:,idx%384,:]
            # img_d, mask_d = self.rotation_crop(img_slice_tmp, mask_slice_tmp, dim=2)   # randomly rotate the slice with (-10,10)
            # img_d = img_r[:,40:280,:]
            # mask_d = mask_r[:,40:280,:]
            # blacks = np.zeros((240,40))
            # img_slice = np.concatenate((blacks, img_slice_tmp, blacks), axis=1)
            # mask_slice = np.concatenate((blacks, mask_slice_tmp, blacks), axis=1)

            # img_slice = img[40:280,60:300,idx%160]
            # mask_slice = mask[40:280,60:300,idx%160]     # crop the slice
        # img_slice_tmp = img[:,:,idx%160]
        # mask_slice_tmp = mask[:,:,idx%160]
        img_d, mask_d = self.rotation_crop(img, mask)   # randomly rotate the slice with (-10,10)
        # img_hpf = DFT(img_d[0,:,:], win=3)
        # img_h = img_hpf[np.newaxis,:,:]
            # img_d = img_r[:,40:280,60:300]
            # mask_d = mask_r[:,40:280,60:300]

        # img_d, mask_d = self.rotation(img_slice, mask_slice)   # randomly rotate the slice with (-10,10)
        
        inv_mask_d = (mask_d - 1) * (-1)

        mask_dist = ndimage.morphology.distance_transform_edt(mask_d)
        inv_mask_dist = ndimage.morphology.distance_transform_edt(inv_mask_d)
        dist = mask_dist + inv_mask_dist

        img_o = img_d[np.newaxis,:,:]
        # img_o2 = img_d2[np.newaxis,:,:]
        # img_o3 = img_d3[np.newaxis,:,:]
        mask_o = mask_d[np.newaxis,:,:]
        # mask_o2 = mask_d2[np.newaxis,:,:]
        # mask_o3 = mask_d3[np.newaxis,:,:]
        dist_o = dist[np.newaxis,:,:]


        return {
            'image': torch.as_tensor(img_o.copy()).float().contiguous(),
            # 'image2': torch.as_tensor(img_o2.copy()).float().contiguous(),
            # 'image3': torch.as_tensor(img_o3.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask_o.copy()).float().contiguous(),
            # 'mask2': torch.as_tensor(mask_o2.copy()).float().contiguous(),
            # 'mask3': torch.as_tensor(mask_o3.copy()).float().contiguous(),
            'dist': torch.as_tensor(dist_o.copy()).float().contiguous()
        }


