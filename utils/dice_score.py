import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon=1):

    # tmp = (input > out_threshold).float()
    inter = torch.dot(input.reshape(-1), target.reshape(-1))
    sets_sum = torch.sum(input) + torch.sum(target)
    dice_coeff = (2 * inter + epsilon) / (sets_sum + epsilon)
    avg_dice = dice_coeff 

    return avg_dice


def dice_loss(input: Tensor, target: Tensor):

    fn = dice_coeff
    return 1 - fn(input, target)
