import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms,  models
from collections import OrderedDict
from PIL import Image
import numpy as np
import json


def load_data(path):
    
    #setting path for training, validation and testing folder
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    #transforms for the training, validation, and testing sets    
    training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                             ])

    validation_or_testing_transforms=transforms.Compose([transforms.Resize(255),
                                                          transforms.CenterCrop(224),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                               [0.229, 0.224, 0.225])
                                                        ])


    #Load the datasets with ImageFolder
    #Loading training dataset 
    train_data = datasets.ImageFolder(train_dir, transform=training_transforms)

    #Loading validation dataset
    validation_data=datasets.ImageFolder(valid_dir, transform=validation_or_testing_transforms)

    #Loading testing dataset
    test_data=datasets.ImageFolder(test_dir, transform=validation_or_testing_transforms)


    
    #Using the image datasets and the transforms, define the dataloaders
    #DataLoader for training set
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    #DataLoader for validation set
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=64)

    #DataLoader for train
    testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    
    return train_data, validation_data, train_dataloader, validation_dataloader


    
    
    
    
    





