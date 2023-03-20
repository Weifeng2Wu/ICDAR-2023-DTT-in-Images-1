import os
import torch
import time
import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Config import CFG
from Model.model import build_model
from Losses.loss import build_loss
from Losses.Focal import FocalLoss
from Datasets.Process import build_dataloader, build_dataset
from Datasets.transform import build_transforms 
from sklearn.model_selection import StratifiedGroupKFold, KFold  


if __name__ == '__main__':
    utils.set_seed(CFG.seed)
    ckpt_path = f"./{CFG.ckpt_fold}/{CFG.ckpt_name}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    train_val_flag = True
    if train_val_flag:
        col_name = ['img_name', 'img_path', 'img_label']
        imgs_info = []  
        for img_name in os.listdir(CFG.tampered_img_paths):
            if img_name.endswith('.jpg'):
                imgs_info.append(["p_" + img_name, os.path.join(CFG.tampered_img_paths, img_name), 1])

        for img_name in os.listdir(CFG.untampered_img_paths):
            if img_name.endswith('.jpg'): 
                imgs_info.append(["n_" + img_name, os.path.join(CFG.untampered_img_paths, img_name), 0])

        imgs_info_array = np.array(imgs_info)
        df = pd.DataFrame(imgs_info_array, columns=col_name)
        
        kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df.loc[val_idx, 'fold'] = fold



        for fold in range(CFG.n_fold):
            print(f'#' * 40, flush=True)
            print(f'###### Fold: {fold}', flush=True)
            print(f'#' * 40, flush=True)

            data_transforms = build_transforms(CFG)
            train_loader, valid_loader = build_dataloader(df, True, fold, data_transforms)  
            model = build_model(CFG, pretrain_flag=True) 
            optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CFG.lr_drop)
            losses_dict = build_loss()
            # losses_dict = FocalLoss()  

            
            best_val_acc = 0
            best_epoch = 0

            acc_list=[]
            epoch_size_list =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

            score_list = []

            for epoch in range(1, CFG.epoch + 1):
                start_time = time.time()
                ###############################################################
                ##### >>>>>>> step3: train & val <<<<<<
                ###############################################################
                utils.train_one_epoch(model, train_loader, optimizer, losses_dict, CFG)
                lr_scheduler.step()
                val_acc = utils.valid_one_epoch(model, valid_loader, CFG)
                acc_list.append(val_acc)
                ###############################################################
                ##### >>>>>>> step4: save best model <<<<<<
                ###############################################################
                is_best = (val_acc > best_val_acc)
                best_val_acc = max(best_val_acc, val_acc)
                if is_best:
                    save_path = f"{ckpt_path}/best_fold{fold}_epoch{epoch}.pth"
                    if os.path.isfile(save_path):
                        os.remove(save_path)
                    torch.save(model.state_dict(), save_path)

                epoch_time = time.time() - start_time
                print("epoch:{}, time:{:.2f}s, best:{:.2f}\n".format(epoch, epoch_time, best_val_acc), flush=True)

            plt.plot(epoch_size_list, acc_list)
            plt.scatter(epoch_size_list, acc_list, c='red')
            plt.title('Training Accuracy'+' Fold:'+str(fold))
            plt.xlabel("epoch", fontdict={'size': 16})
            plt.ylabel("acc", fontdict={'size': 16})

        plt.show()

    test_flag = False
    if test_flag:

        col_name = ['img_name', 'img_path', 'pred_prob']

        imgs_info = []  
        test_imgs = os.listdir(CFG.test_img_path)
        test_imgs.sort(key=lambda x: x[:-4]) 
        for img_name in test_imgs:
            if img_name.endswith('.jpg'): 
                imgs_info.append([img_name, os.path.join(CFG.test_img_path, img_name), 0])

        imgs_info_array = np.array(imgs_info)
        test_df = pd.DataFrame(imgs_info_array, columns=col_name)

        data_transforms = build_transforms(CFG)
        test_loader = build_dataloader(test_df, False, None, data_transforms) 

        ###############################################################
        ##### >>>>>>> step3: test <<<<<<
        ###############################################################
        ckpt_paths = [CFG.ckpt]  
        test_df = utils.test_one_epoch(test_df, ckpt_paths, test_loader, CFG)
        submit_df = test_df.loc[:, ['img_name', 'pred_prob']]
        submit_df.to_csv(CFG.submit, header=False, index=False, sep=' ')