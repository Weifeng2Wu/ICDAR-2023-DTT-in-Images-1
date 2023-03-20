from tqdm import tqdm
import torch
import random
import numpy as np
import pandas as pd
from Config import CFG
from torch.cuda import amp
from Model.model import build_model

def set_seed(seed=42):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # pytorch
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train_one_epoch(model, train_loader, optimizer, losses_dict, CFG):
    model.train()
    scaler = amp.GradScaler()
    losses_all, ce_all = 0, 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    # p = pbar[0]
    for _, (images, gt) in pbar:
        optimizer.zero_grad()

        images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]
        gt = gt.to(CFG.device)

        with amp.autocast(enabled=True):
            y_preds = model(images)
            ce_loss = losses_dict["CELoss"](y_preds, gt.long())
            losses = ce_loss


        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        losses_all += losses.item() / images.shape[0]
        ce_all += ce_loss.item() / images.shape[0]

    current_lr = optimizer.param_groups[0]['lr']
    print("lr: {:.4f}".format(current_lr), flush=True)
    print("loss: {:.3f}, ce_all: {:.3f}".format(losses_all, ce_all), flush=True)


@torch.no_grad()
def valid_one_epoch(model, valid_loader, CFG):
    model.eval()
    df = pd.DataFrame( columns=['label','value'])

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc='Valid ')
    for _, (images, gt) in pbar:
        images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]

        gt = gt.to(CFG.device)
        list_label = gt.cpu().numpy().tolist()
        y_preds = model(images)
        prob = torch.nn.functional.softmax(y_preds, dim=-1)[:, 1].detach().cpu().numpy().tolist()
        res_array = [list_label, prob]
        res_array = np.array(res_array).T
        est_df = pd.DataFrame(res_array, columns=['label', 'value'])
        est_df.label = est_df.label.astype('int')
        df = pd.concat([df,est_df],ignore_index=True)

    pred_untampers = df.query('label==0')
    pred_tampers = df.query('label==1')
    thres = np.percentile(pred_untampers.values[:, 1], np.arange(90, 100, 1))
    recall = np.mean(np.greater(pred_tampers.values[:, 1][:, np.newaxis], thres).mean(axis=0))
    print("recall: {:.2f}".format(recall * 100), flush=True)

    return recall * 100



@torch.no_grad()
def test_one_epoch(test_df, ckpt_paths, test_loader, CFG):
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Test: ')
    for _, (images, ids) in pbar:

        images = images.to(CFG.device, dtype=torch.float)  # [b, c, w, h]

        ############################################
        ##### >>>>>>> cross validation infer <<<<<<
        ############################################

        for sub_ckpt_path in ckpt_paths:
            model = build_model(CFG, pretrain_flag=True)  # just dummy code
            model.load_state_dict(torch.load(sub_ckpt_path))
            model.eval()
            y_preds = model(images)  # [b, c, w, h]
            prob = torch.nn.functional.softmax(y_preds, dim=-1)[:, 1].detach().cpu().numpy()
            test_df.loc[test_df['img_name'].isin(ids), 'pred_prob'] = prob

    return test_df

