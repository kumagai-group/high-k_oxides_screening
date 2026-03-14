import sys
import torch
from model.params import DatasetParams
from model.model import SitePooling
from common.utils import AverageMeter, to_gpu


def train(
    model,
    train_loader,
    optimizer,
    scheduler,
    loss_func,
    params_data: DatasetParams, 
    min_lr=1e-5,
    c='cuda', 
):
    losses = AverageMeter()
    model.train()

    pooling = SitePooling()
    pooling.eval()

    for batch_data in train_loader:
        y = batch_data.prop

        mask = batch_data.pooling_mask

        optimizer.zero_grad()

        data_list = batch_data.to_data_list()
        pred = model(data_list)

        y = to_gpu(y, c=c)
        # y = get_same_shape(y, pred.shape)

        if params_data.target == "born":
            loss = loss_func(pooling(pred, mask), pooling(y, mask))
        else:
            loss = loss_func(pred, y)
        loss.backward()
        losses.update(loss.item())

        if torch.any(torch.isnan(loss)):
            print("Encountered NaN in loss!")
            sys.exit(1)

        optimizer.step()
        scheduler.step()

        for param_group in optimizer.param_groups:
            if param_group['lr'] < min_lr:
                param_group['lr'] = min_lr

    return losses.avg


def validate(
    model,
    valid_loader,
    loss_func,
    params_data: DatasetParams, 
    is_test=False,
    c='cuda', 
):
    losses = AverageMeter()

    pooling = SitePooling()
    pooling.eval()

    model.eval()
    matname_all, true_all, pred_all = [], [], []
    for batch_data in valid_loader:
        matnames = batch_data.matname

        with torch.no_grad():
            data_list = batch_data.to_data_list()
            pred = model(data_list)

        y = batch_data.prop
        
        mask = batch_data.pooling_mask

        y = to_gpu(y, c=c)
        # y = get_same_shape(y, pred.shape)

        if params_data.target == "born":
            loss = loss_func(pooling(pred, mask), pooling(y, mask))
        else:
            loss = loss_func(pred, y)
        losses.update(loss.item())

        if is_test:
            matname_all.extend(matnames), 
            true_all.append(y.detach().cpu()), 
            pred_all.append(pred.detach().cpu()), 
    
    dict_valid = {
        "matname": matname_all, 
        "true": true_all, 
        "pred": pred_all, 
    }

    return losses.avg, dict_valid
