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
import prepare_dataset
import model_creation



parser = argparse.ArgumentParser(
    description = 'Parser for train.py'
)


parser.add_argument('--data_dir', action="store", default="./flowers/")
parser.add_argument('--save_dir', action="store", default="./checkpoint.pth")
parser.add_argument('--arch', action="store", default="densenet121")
parser.add_argument('--learning_rate', action="store", type=float,default=0.003)
parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=256)
parser.add_argument('--epochs', action="store", default=2, type=int)
parser.add_argument('--dropout', action="store", type=float, default=0.2)
parser.add_argument('--gpu', action="store", default="gpu")
args = parser.parse_args()

device = torch.device("cuda" if args.gpu=="gpu" else "cpu")


def main():
    
    #prepare training, validation, testing datasets
    train_data, validation_data, train_dataloader, validation_dataloader = prepare_dataset.load_data(args.data_dir)
    print('Dataset loading done')
    
    
    print('creating model...')
    #create the model
    model= model_creation.create_model(args.arch, args.hidden_units, args.dropout)
    print('model created')
    
    
    print('training the model...')
    #training the model
    model.to(device);
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    epochs = args.epochs
    train_losses, validation_losses = [], []
    for e in range(epochs):

        running_loss = 0
        i=0
        for inputs, labels in train_dataloader:

            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        else:
            validation_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validation_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    validation_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


            train_losses.append(running_loss/len(train_dataloader))
            validation_losses.append(validation_loss/len(validation_dataloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_dataloader)),
                  "Validation Loss: {:.3f}.. ".format(validation_loss/len(validation_dataloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validation_dataloader)))

            model.train()
       
    print('training completed')
    
    print('saving the model..')
    #save the checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch': args.arch,
                  'hidden_units':args.hidden_units,
                  'dropout':args.dropout,
                  'state_dict': model.state_dict(), 
                  'class_to_idx': model.class_to_idx}
   
    torch.save(checkpoint, args.save_dir)
    print("model is saved")
    

if __name__== "__main__":
    main()
    
