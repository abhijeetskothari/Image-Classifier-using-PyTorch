import pandas as pd
import torch
from torch import nn, optim
from torchvision import datasets, transforms,  models
from collections import OrderedDict
from PIL import Image
import seaborn as sns
import numpy as np
import json
import argparse



parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)


parser.add_argument('--image_path', action="store", default= "flowers/test/10/image_07090.jpg")
parser.add_argument('--checkpoint_path', action="store", default="./checkpoint.pth")
parser.add_argument('--top_K', action="store", type=int, default=5)
parser.add_argument('--category_names', action="store", default="./cat_to_name.json")
parser.add_argument('--gpu', action="store", default="gpu")
args = parser.parse_args()
device = torch.device("cuda" if args.gpu=="gpu" else "cpu")



def load_model(checkpoint_path):
    
    chkpt = torch.load(checkpoint_path)
    
    arch_options = {"vgg16":25088, "densenet121":1024}
    arch=chkpt['arch']
    hidden_units=chkpt['hidden_units']
    dropout=chkpt['dropout']
      
    if arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        
    for param in model.parameters():
            param.requires_grad = False
    
    model.class_to_idx = chkpt['class_to_idx']

    # Create the classifier
    classifier = nn.Sequential(nn.Linear(arch_options[arch], hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(hidden_units, 102),
                                       nn.LogSoftmax(dim=1))

    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(chkpt['state_dict'])
    return model    

def process_image(img):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
        
        
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def predict(image, model, k=args.top_K):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # Process image
    img = process_image(image) 
    img = torch.from_numpy(img).type(torch.FloatTensor) 
    img.unsqueeze_(0)
    
    # Predict top k
    model.eval()
    probs = torch.exp(model.forward(img)) 
    top_probs, top_labs = probs.topk(k) 
    top_probs = top_probs.detach().numpy().tolist()[0] 
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels, top_flowers  
    
    
if __name__== "__main__":
    
    #load model
    model=load_model(args.checkpoint_path)
    
    #predict_image
    image=Image.open(args.image_path)
    probs, labs, flowers = predict(image, model)
    print()
    for fl, pb in zip(flowers, probs):
        print("flower name= {}, probablity={:3f}".format(fl, pb))
