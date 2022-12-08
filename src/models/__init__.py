import torchvision
from .add_gcn import ADD_GCN
import torch
# from .vit_base import VisionTransformer, CONFIGS
# from .vit_mcar import VisionTransformer, CONFIGS
from .vit_multirams import VisionTransformer, CONFIGS
# from .vit_csra import VisionTransformer, CONFIGS
import numpy as np

model_dict = {'ADD_GCN': ADD_GCN}

def get_model(num_classes, args):
    res101 = torchvision.models.resnet101(pretrained=True)
    model = model_dict[args.model_name](res101, num_classes)
    return model


def get_vit(num_classes):
    conf = CONFIGS['ViT-B_16']
    model = VisionTransformer(conf, 448, zero_head=True, num_classes=80, smoothing_value=0.0)
    model.load_from(np.load("Your path to pretrained model"))
    
    return model

