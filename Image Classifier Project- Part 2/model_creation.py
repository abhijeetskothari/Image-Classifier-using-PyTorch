import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms,  models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json


def create_model(arch, hidden_units, drop_rate):
    
    arch_options = {"vgg16":25088, "densenet121":1024}
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
   
    
    #freezing paramenter so that we do not backdrop through them
    for param in model.parameters():
        param.requires_grad = False
        
    #create a feed forward model 
    classifier= nn.Sequential(nn.Linear(arch_options[arch], hidden_units),
                              nn.ReLU(),
                              nn.Dropout(drop_rate),
                              nn.Linear(hidden_units, 102),
                              nn.LogSoftmax(dim=1)
                             )

    #changing the classifier of pre-trained model with our feed-forward classifier
    model.classifier=classifier
    return model