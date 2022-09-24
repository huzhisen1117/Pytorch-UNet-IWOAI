import torch


def dist_loss(pred, true, dist_map):
    
    pred_m = (pred > 0.5).float()             # binary

    diff1 = (pred_m > true).float()      # difference region
    mult1 = diff1 * pred                      
    loss1 = mult1 * dist_map
    
    diff2 = (pred_m < true).float()
    mult2 = diff2 * (1 - pred)
    loss2 = mult2 * dist_map
    loss = loss1 + loss2

    # diff = (pred_m != true).float()
    # mult = diff * ((pred-true) ** 2)
    # loss = mult * dist_map

    return loss.sum() 
