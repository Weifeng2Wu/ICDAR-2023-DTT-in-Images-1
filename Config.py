import torch

class CFG:
        # step1: hyper-parameter
        seed = 42
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        ckpt_fold = "output"
        ckpt_name = "efficientnetb4_img512512_bs8"  
        ckpt =  "output/efficientnetb4_img512512_bs8/best_fold2_epoch15.pth"
        tampered_img_paths = "data/train/tampered/imgs"
        untampered_img_paths = "data/train/untampered/"
        test_img_path = "data/test/imgs"
        submit = "submit_dummy_b4_4.csv"
        
        # step2: data
        n_fold = 4
        img_size = [512, 512]
        train_bs = 10
        valid_bs = train_bs * 2
        # step3: model
        backbone = 'efficientnet_b4'
        # backbone = 'efficientnet_b0'
        num_classes = 2
        # step4: optimizer
        epoch = 15
        lr = 2e-3
        wd = 1e-5
        lr_drop = 8
        # step5: infer
        thr = 0.5