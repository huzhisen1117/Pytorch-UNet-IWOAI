import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import dice_coeff


def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']   # , batch['edge']
        # image2 = torch.cat((image,edge), dim=1)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.float32)
        # edge = edge.to(device=device, dtype=torch.float32)
        # image2 = image2.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # predict the mask
            mask_tmp = net(image)
            mask_pred = (mask_tmp>0.5).float()  # threshold = 0.5
            # compute the Dice score, ignoring background
            dice_score += dice_coeff(mask_pred, mask_true)

           

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches
