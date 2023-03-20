import timm
import torch
from Config import CFG
from Model.model_effv2 import efficientnetv2_m 

def build_model(CFG, pretrain_flag=False):
    if pretrain_flag:
        pretrain_weights = "imagenet"
    else:
        pretrain_weights = False
        
    model = timm.create_model(CFG.backbone, pretrained=pretrain_flag, num_classes=CFG.num_classes)
    
    # model = efficientnetv2_m(num_classes=2).to(CFG.device)
    # weights_dict = torch.load(PRE_MODEL_PATH, map_location=CFG.device)
    # load_weights_dict = {k: v for k, v in weights_dict.items()
    #                      if model.state_dict()[k].numel() == v.numel()}
    # model.load_state_dict(load_weights_dict, strict=False)
    model.to(CFG.device)
    return model

