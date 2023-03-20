import torch

def build_loss():
    CELoss = torch.nn.CrossEntropyLoss()
    return {"CELoss": CELoss}