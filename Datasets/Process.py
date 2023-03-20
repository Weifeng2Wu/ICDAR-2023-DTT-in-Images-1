import cv2
import torch
import numpy as np
from Config import CFG
from torch.utils.data import Dataset, DataLoader

class build_dataset(Dataset):
    def __init__(self, df, train_val_flag=True, transforms=None):

        self.df = df
        self.train_val_flag = train_val_flag  #
        self.img_paths = df['img_path'].tolist()
        self.ids = df['img_name'].tolist()
        self.transforms = transforms

        if train_val_flag:
            self.label = df['img_label'].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        id = self.ids[index]
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # [h, w, c]

        if self.train_val_flag:  #train
            data = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1))  # [c, h, w]
            gt = self.label[index]
            return torch.tensor(img), torch.tensor(int(gt))

        else:  # test
            data = self.transforms(image=img)
            img = np.transpose(data['image'], (2, 0, 1))  # [c, h, w]
            return torch.tensor(img), id

def build_dataloader(df, train_val_flag=True, fold=None, data_transforms=None):
    if train_val_flag:
        train_df = df.query("fold!=@fold").reset_index(drop=True)
        valid_df = df.query("fold==@fold").reset_index(drop=True)

        train_dataset = build_dataset(train_df, train_val_flag=train_val_flag, transforms=data_transforms['train'])
        valid_dataset = build_dataset(valid_df, train_val_flag=train_val_flag, transforms=data_transforms['valid_test'])

        train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=True, pin_memory=True,
                                  drop_last=False)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs, num_workers=0, shuffle=False, pin_memory=True)

        return train_loader, valid_loader

    else:
        test_dataset = build_dataset(df, train_val_flag=train_val_flag, transforms=data_transforms['valid_test'])
        test_loader = DataLoader(test_dataset, batch_size=CFG.train_bs, num_workers=0, shuffle=False, pin_memory=True,
                                 drop_last=False) 
        return test_loader