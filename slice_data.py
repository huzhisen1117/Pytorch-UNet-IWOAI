import h5py
from os import listdir
from os.path import splitext
from pathlib import Path

dir_img = './val/imgs/'
dir_seg = './val/masks/'
save_img = './sliceval/imgs/'
save_seg = './sliceval/masks/'

filelist = [splitext(file)[0] for file in listdir(dir_img) if not file.startswith('.')]

for i in range(len(filelist)):
    img_file = list(Path(dir_img).glob(filelist[i] + '.im'))
    seg_file = list(Path(dir_seg).glob(filelist[i] + '.seg'))
    with h5py.File(img_file[0],'r') as f:
        img = f['data'][:]
    with h5py.File(seg_file[0],'r') as f:
        seg = f['data'][:]
    for j in range(img.shape[2]):
        imgs = img[:,:,j]
        segs = seg[:,:,j,:]
        if segs.sum() > 0:
            save_name_img = save_img + filelist[i] + '_slice{}.im'.format(j)
            save_name_seg = save_seg + filelist[i] + '_slice{}.seg'.format(j)
            with h5py.File(save_name_img,'w') as h5f:
                h5f.create_dataset('data',data=imgs)
            with h5py.File(save_name_seg,'w') as h5f:
                h5f.create_dataset('data',data=segs)
